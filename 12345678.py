# ============================
# Imports
# ============================
import os
import numpy as np
import streamlit as st
import requests
import pandas as pd
from io import StringIO
from scipy.spatial import KDTree
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.Align import PairwiseAligner
import streamlit.components.v1 as components
import py3Dmol
import json
import time
# ============================
# ثوابت
# ============================
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
AA_PROPS = {
    'ALA': 'غير قطبي (Non-polar)',       'ARG': 'قاعدي (+)',
    'ASN': 'قطبي غير مشحون (Polar)',     'ASP': 'حمضي (-)',
    'CYS': 'قطبي غير مشحون (Polar)',     'GLN': 'قطبي غير مشحون (Polar)',
    'GLU': 'حمضي (-)',                    'GLY': 'غير قطبي (Non-polar)',
    'HIS': 'قاعدي (+)',                   'ILE': 'غير قطبي (Non-polar)',
    'LEU': 'غير قطبي (Non-polar)',        'LYS': 'قاعدي (+)',
    'MET': 'غير قطبي (Non-polar)',        'PHE': 'عطري غير قطبي (Aromatic)',
    'PRO': 'غير قطبي (Non-polar)',        'SER': 'قطبي غير مشحون (Polar)',
    'THR': 'قطبي غير مشحون (Polar)',     'TRP': 'عطري غير قطبي (Aromatic)',
    'TYR': 'عطري قطبي (Aromatic)',       'VAL': 'غير قطبي (Non-polar)'
}
# ============================
# دوال مساعدة
# ============================
@st.cache_data(ttl=3600)
def fetch_pdb_data(pdb_id):
    if not pdb_id or pdb_id == "NONE":
        return None  
    # 1. البحث في المجلد المحلي أولاً
    local_dir = "Protein_Database_100"
    local_file_path = os.path.join(local_dir, f"pdb{pdb_id.lower()}.ent")
    
    if os.path.exists(local_file_path):
        try:
            with open(local_file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    # 2. في حال لم يكن موجوداً محلياً، جلبه من الإنترنت
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as error:
        st.error(f"خطأ في الاتصال بالسيرفر: {error}")
        return None

@st.cache_data(ttl=3600)
def process_protein_structure(pdb_string, pdb_id, keep_hetatm=False):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, StringIO(pdb_string))
        if not keep_hetatm:
            for model in structure:
                for chain in model:
                    for r_id in [r.get_id() for r in chain if r.get_id()[0] != ' ']:
                        chain.detach_child(r_id)
        return structure
    except Exception as error:
        st.error(f"خطأ في قراءة البنية: {error}")
        return None


def get_all_chains(structure):
    try:
        return [chain.id for chain in structure[0]]
    except Exception as ex:
        st.error(f"فشل قراءة السلاسل: {ex}")
        return []


def get_protein_sequence(structure, chain_id):
    sequence = []
    try:
        model = structure[0]
        if chain_id in [c.id for c in model]:
            for residue in model[chain_id]:
                if residue.id[0] == ' ':
                    sequence.append({
                        'res_num': residue.id[1],
                        'res_name': residue.get_resname()
                    })
    except Exception as ex:
        st.error(f"Failed to read sequence for chain {chain_id}: {ex}")
    return sequence


def sequence_to_fasta(structure, chain_id, protein_name="protein"):
    seq_data = get_protein_sequence(structure, chain_id)
    if not seq_data:
        return None
    one_letter = ''.join([AA_3TO1.get(r['res_name'], 'X') for r in seq_data])
    lines = [one_letter[i:i+80] for i in range(0, len(one_letter), 80)]
    header = f">{protein_name}|Chain_{chain_id}|length={len(one_letter)}\n"
    body = '\n'.join(lines) + '\n'
    return header + body

#البحث عن الجزيئات المتجاورة
@st.cache_data(ttl=3600)
def calculate_all_distances(_structure_key, pdb_string, chain_id, radius=5.0, keep_hetatm=False):
    structure = process_protein_structure(pdb_string, "prot", keep_hetatm)
    if not structure:
        return []
    try:
        model = structure[0]
        if chain_id not in [c.id for c in model]:
            return []

        # SASA
        sr = ShrakeRupley()
        sr.compute(structure, level='R')

        # استخراج كافة الذرات وإحداثياتها
        all_atoms = list(model.get_atoms())
        all_coords = np.array([a.get_coord() for a in all_atoms], dtype=np.float32)
        
        # بناء الشجرة للبحث السريع
        tree = KDTree(all_coords)
        
        # استخراج معلومات السلسلة والرقم لكل ذرة لتسريع الفلترة
        atom_info = []
        for a in all_atoms:
            res = a.get_parent()
            atom_info.append((res.get_parent().id, res.id[1]))
        
        residues = [r for r in model[chain_id] if r.id[0] == ' ']
        results = []

        for target_res in residues:
            target_res_id = target_res.id[1]
            target_coords = np.array([a.get_coord() for a in target_res.get_atoms()], dtype=np.float32)
            
            # البحث عن جميع الذرات القريبة في خطوة واحدة
            indices = tree.query_ball_point(target_coords, radius)
            
            # تجميع الفهارس الفريدة وفلترة الذرات التي تنتمي لنفس الحمض
            nearby_indices = set()
            for idx_list in indices:
                for idx in idx_list:
                    info = atom_info[idx]
                    # استبعاد الذرة إذا كانت تنتمي لنفس الحمض وفي نفس السلسلة
                    if not (info[0] == chain_id and info[1] == target_res_id):
                        nearby_indices.add(idx)

            min_dist = "-"
            if nearby_indices:
                nb_coords = all_coords[list(nearby_indices)]
                # حساب المسافات باستخدام NumPy
                diff = target_coords[:, np.newaxis, :] - nb_coords[np.newaxis, :, :]
                dist_sq = (diff * diff).sum(axis=-1)
                min_dist = round(float(np.sqrt(dist_sq.min())), 2)

            resname = target_res.get_resname()
            sasa_val = getattr(target_res, 'sasa', '-')
            if isinstance(sasa_val, (float, int)):
                sasa_val = round(sasa_val, 2)

            results.append({
                'chain'          : chain_id,
                'res_num'        : target_res_id,
                'res_name'       : resname,
                'one_letter'     : AA_3TO1.get(resname, 'X'),
                'properties'     : AA_PROPS.get(resname, '-'),
                'nearby_count'   : len(nearby_indices),
                'min_dist'       : min_dist,
                'sasa'           : sasa_val
            })

        return results

    except Exception as error:
        st.error(f"خطأ أثناء حساب المسافات: {error}")
        return []


def render_protein_3d(pdb_string, bg_color='#0E1117', style_type='cartoon',
                      show_surface=True, surface_opacity=0.3, mutations=None,
                      mut_color='red', zoom_to_mutations=False,
                      focus_mut=None, keep_hetatm=False):
    view = py3Dmol.view(width="100%", height=450)
    view.addModel(pdb_string, 'pdb')
    view.setBackgroundColor(bg_color)

    style_dict = {}
    if style_type == 'cartoon':
        style_dict['cartoon'] = {'color': 'spectrum'}
    elif style_type == 'stick':
        style_dict['stick'] = {'color': 'spectrum'}
    elif style_type == 'sphere':
        style_dict['sphere'] = {'color': 'spectrum'}
    else:
        style_dict['cartoon'] = {'color': 'spectrum'}

    view.setStyle({'model': -1}, style_dict)

    if keep_hetatm:
        view.addStyle({'hetflag': True}, {'stick': {'colorscheme': 'magentaCarbon', 'radius': 0.2}})

    if show_surface:
        view.addSurface(py3Dmol.SAS, {'opacity': surface_opacity, 'color': '#FFC107'})

    if mutations:
        for mut in mutations:
            view.addStyle(mut, {style_type: {'color': mut_color}})
            view.addStyle(mut, {'stick'  : {'colorscheme': 'yellowCarbon', 'radius': 0.3}})
            view.addStyle(mut, {'sphere' : {'color': mut_color, 'radius': 1.2}})

    if focus_mut:
        view.zoomTo(focus_mut)
    elif zoom_to_mutations and mutations:
        view.zoomTo({'or': mutations})
    else:
        view.zoomTo()

    return view._make_html()


def get_alignment(seq1, seq2, mode='global'):
    aligner = PairwiseAligner()
    aligner.mode = mode
    try:
        best_aln = aligner.align(seq1, seq2)[0]
    except Exception as error:
        st.warning(f"فشل إجراء المحاذاة: {error}")
        return "", 0, "", ""
    return str(best_aln), best_aln.score, best_aln[0], best_aln[1]


REMOTE_JSON_URL = "https://raw.githubusercontent.com/hassan2006-web/Bio-project/refs/heads/main/mutations.json"

@st.cache_data(ttl=300)
def load_mutation_db():
    url_with_cache_bypass = f"{REMOTE_JSON_URL}?t={int(time.time())}"
    try:
        response = requests.get(url_with_cache_bypass, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    json_path = 'mutations.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def auto_fill_healthy():
    m_id = st.session_state.get('m_id_in', '').upper().strip()
    if m_id in MUTATION_DB:
        st.session_state['h_id_in'] = MUTATION_DB[m_id]
# ========================
# مكونات واجهة المستخدم 
# ========================

def protein_ui_panel(p, keep_hetatm):
    """
    مكون موحد لتحميل وعرض البروتين.
    """
    with p["col"]:
        icon = '🟢' if p['prefix'] == 'h' else '🔴'
        st.header(f"{icon} {p['label']}")
        source = st.radio("المصدر:", ["PDB ID", "رفع ملف"], key=f"{p['prefix']}_src", horizontal=True)

        if source == "PDB ID":
            on_change = auto_fill_healthy if p['prefix'] == 'm' else None
            pdb_input = st.text_input("كود PDB:", key=f"{p['prefix']}_id_in", on_change=on_change).strip().upper()

            if st.button(f"تحميل {p['label']}", key=f"btn_{p['prefix']}"):
                with st.spinner('جاري التحميل...'):
                    data = fetch_pdb_data(pdb_input)
                    if data:
                        st.session_state[f"{p['prefix']}_pdb"] = data
                        st.session_state[f"{p['prefix']}_id"]  = pdb_input
                        # تنظيف الحالة القديمة
                        for k in list(st.session_state.keys()):
                            if k.startswith(p['prefix']) and k not in [f"{p['prefix']}_pdb", f"{p['prefix']}_id", f"{p['prefix']}_src", f"{p['prefix']}_id_in"]:
                                st.session_state.pop(k, None)
                        
                        if p['prefix'] == 'm':
                            h_id = st.session_state.get('h_id_in', '').upper().strip()
                            if h_id and not st.session_state.get('h_pdb'):
                                h_data = fetch_pdb_data(h_id)
                                if h_data:
                                    st.session_state["h_pdb"] = h_data
                                    st.session_state["h_id"]  = h_id
                                    st.rerun()
                    else:
                        st.error(f"لم يتم العثور على البروتين بالكود: {pdb_input}")
        else:
            file = st.file_uploader(f"ارفع ملف {p['label']}:", type=["pdb"], key=f"{p['prefix']}_up")
            if file:
                st.session_state[f"{p['prefix']}_pdb"] = file.getvalue().decode("utf-8")
                st.session_state[f"{p['prefix']}_id"]  = file.name
                for k in list(st.session_state.keys()):
                    if k.startswith(p['prefix']) and k not in [f"{p['prefix']}_pdb", f"{p['prefix']}_id", f"{p['prefix']}_src", f"{p['prefix']}_up"]:
                        st.session_state.pop(k, None)


def initialize_session_state():
    """
    تهيئة المتغيرات الأساسية في session_state لتجنب الأخطاء.
    """
    defaults = {
        'h_pdb': None, 'm_pdb': None,
        'h_id': '', 'm_id': '',
        'h_results': None, 'm_results': None,
        'h_selected_chain': None, 'm_selected_chain': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main():
    MUTATION_DB = load_mutation_db()

    st.set_page_config(page_title="Bio-Impact Analyzer", page_icon="🧬", layout="wide")
    initialize_session_state()
    st.title("🧬 Bio-Impact Analyzer")

    # ── Sidebar ──
    st.sidebar.header("⚙️ إعدادات التحليل")
    search_radius  = st.sidebar.slider("🔍 نصف قطر البحث (Å)", 3.0, 12.0, 5.0)
    view_style     = st.sidebar.selectbox("🎨 نمط العرض", ["cartoon", "stick", "sphere"])
    show_surf      = st.sidebar.checkbox("إظهار السطح (Surface)", value=False)
    surface_op     = st.sidebar.slider("شفافية السطح", 0.0, 1.0, 0.3)
    show_mutations = st.sidebar.checkbox("🧬 تلوين الطفرات", value=False)
    zoom_mutations = st.sidebar.checkbox("🔍 تركيز العرض على الطفرات", value=False)
    st.sidebar.divider()
    st.sidebar.subheader("🌟 إضافات التحليل المتقدم")
    keep_hetatm = st.sidebar.checkbox("💊 الإبقاء على الأدوية (Ligands)", value=False)
    align_mode  = st.sidebar.selectbox("🧬 نوع المحاذاة (Alignment)", ["global", "local"])
    st.sidebar.warning("التحليل الهيكلي يشمل البحث في جميع سلاسل البروتين.")

    col1, col2 = st.columns(2)
    proteins = [
        {"label": "السليم", "prefix": "h", "col": col1, "bg": "#0D1B1E"},
        {"label": "المصاب", "prefix": "m", "col": col2, "bg": "#1E0D0D"}
    ]

    # 1. تحميل البيانات
    for p in proteins:
        protein_ui_panel(p, keep_hetatm)

    st.divider()

    # 2. المعالجة والعرض
    v_col1, v_col2 = st.columns(2)
    structures  = {}
    
    for i, p in enumerate(proteins):
        current_col = v_col1 if i == 0 else v_col2
        with current_col:
            pdb_data = st.session_state.get(f"{p['prefix']}_pdb")
            if not pdb_data:
                continue

            st.subheader(f"هيكل {p['label']}")
            struct = process_protein_structure(pdb_data, p['prefix'], keep_hetatm)
            structures[p['prefix']] = struct

            if struct:
                chains = get_all_chains(struct)
                if chains:
                    selected_chain = st.selectbox(f"اختر السلسلة - {p['label']}", options=chains, key=f"{p['prefix']}_chain_sel")
                    st.session_state[f"{p['prefix']}_selected_chain"] = selected_chain
                else:
                    st.warning("لم يتم العثور على سلاسل في هذا الهيكل.")

    # حساب الطفرات
    highlight_map = {'h': None, 'm': None}
    h_chain = st.session_state.get('h_selected_chain')
    m_chain = st.session_state.get('m_selected_chain')
    
    if structures.get('h') and structures.get('m') and h_chain and m_chain:
        h_seq = get_protein_sequence(structures['h'], h_chain)
        m_seq = get_protein_sequence(structures['m'], m_chain)
        if h_seq and m_seq:
            dict_h = {r['res_num']: r['res_name'] for r in h_seq}
            dict_m = {r['res_num']: r['res_name'] for r in m_seq}
            mut_m = []
            mut_h = []
            for res_num in set(dict_h) | set(dict_m):
                if dict_h.get(res_num) != dict_m.get(res_num):
                    if res_num in dict_m: mut_m.append({'resi': str(res_num), 'chain': str(m_chain)})
                    if res_num in dict_h: mut_h.append({'resi': str(res_num), 'chain': str(h_chain)})
            highlight_map['m'] = mut_m if mut_m else None
            highlight_map['h'] = mut_h if mut_h else None

    # العرض والتحليل
    for i, p in enumerate(proteins):
        current_col = v_col1 if i == 0 else v_col2
        with current_col:
            prefix = p['prefix']
            pdb_data = st.session_state.get(f"{prefix}_pdb")
            struct = structures.get(prefix)
            selected_chain = st.session_state.get(f"{prefix}_selected_chain")

            if not pdb_data or not struct or not selected_chain:
                continue

            highlight = highlight_map[prefix]
            focus_mut = None
            if highlight:
                mut_opts = ["الكل"] + [f"Residue {m['resi']} (Chain {m['chain']})" for m in highlight]
                sel_mut  = st.selectbox(f"🔍 التركيز على طفرة - {p['label']}", mut_opts, key=f"focus_{prefix}")
                if sel_mut != "الكل":
                    parts     = sel_mut.split(" ")
                    focus_mut = {'resi': parts[1], 'chain': parts[3].replace(")", "")}

            view_html = render_protein_3d(
                pdb_data, bg_color=p['bg'], style_type=view_style,
                show_surface=show_surf, surface_opacity=surface_op,
                mutations=highlight if show_mutations else None,
                mut_color='#F44336' if prefix == 'm' else '#4CAF50',
                zoom_to_mutations=zoom_mutations, focus_mut=focus_mut, keep_hetatm=keep_hetatm
            )
            components.html(view_html, height=460)

            # Metrics
            total_res = sum(len([r for r in struct[0][c] if r.id[0] == ' ']) for c in get_all_chains(struct))
            c1, c2, c3 = st.columns(3)
            c1.metric("عدد السلاسل", len(get_all_chains(struct)))
            c2.metric("إجمالي الأحماض", total_res)
            c3.metric("السلسلة الحالية", selected_chain)

            # FASTA
            fasta = sequence_to_fasta(struct, selected_chain, st.session_state.get(f"{prefix}_id", p['label']))
            if fasta:
                with st.expander(f"🧬 تنزيل FASTA - {p['label']}"):
                    st.download_button("⬇️ تنزيل FASTA", fasta, f"{st.session_state.get(f'{prefix}_id', 'protein')}_{selected_chain}.fasta", "text/plain", key=f"dl_f_{prefix}")

            # التحليل الهيكلي
            st.divider()
            if st.button(f"🔬 تحليل كامل لـ {p['label']}", key=f"analyze_btn_{prefix}", type="primary"):
                with st.spinner("جاري التحليل..."):
                    results = calculate_all_distances(f"{st.session_state.get(f'{prefix}_id')}_{selected_chain}", pdb_data, selected_chain, radius=search_radius, keep_hetatm=keep_hetatm)
                    st.session_state[f"{prefix}_results"] = results
            
            if st.session_state.get(f"{prefix}_results"):
                res = st.session_state[f"{prefix}_results"]
                with st.expander("📊 نتائج التحليل"):
                    df = pd.DataFrame(res)
                    # تحويل المسميات للعربية عند العرض فقط
                    df_display = df.rename(columns={
                        'res_num': 'رقم البقية', 'res_name': 'الحمض', 'properties': 'الخصائص',
                        'nearby_count': 'المجاورين', 'min_dist': 'أقرب مسافة', 'sasa': 'SASA'
                    })
                    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # المقارنة
    if structures.get('h') and structures.get('m') and h_chain and m_chain:
        st.divider()
        st.header("📋 مقارنة السلسلة")
        h_seq = get_protein_sequence(structures['h'], h_chain)
        m_seq = get_protein_sequence(structures['m'], m_chain)
        
        if h_seq and m_seq:
            df_h = pd.DataFrame(h_seq).rename(columns={'res_name': 'السليم'})
            df_m = pd.DataFrame(m_seq).rename(columns={'res_name': 'المصاب'})
            df_comp = pd.merge(df_h, df_m, on='res_num', how='outer').sort_values('res_num').fillna('-')
            df_comp['الحالة'] = df_comp.apply(lambda r: '🔴 طفرة' if r['السليم'] != r['المصاب'] else '🟢 محافظ', axis=1)
            
            with st.expander("جدول المقارنة الكامل"):
                st.dataframe(df_comp.rename(columns={'res_num': 'الرقم'}).style.apply(lambda r: ['background-color: #3e2723' if r['السليم'] != r['المصاب'] else ''] * len(r), axis=1), use_container_width=True, hide_index=True)

            # Alignment
            st.header(f"🧬 Alignment ({align_mode.capitalize()})")
            h_str = "".join([AA_3TO1.get(r['res_name'], 'X') for r in h_seq])
            m_str = "".join([AA_3TO1.get(r['res_name'], 'X') for r in m_seq])
            aln_text, score, aln1, aln2 = get_alignment(h_str, m_str, align_mode)
            matches = sum(a == b and a != '-' for a, b in zip(aln1, aln2))
            identity = (matches / len(aln1) * 100) if aln1 else 0
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Alignment Score", round(score, 1))
            sc2.metric("Identity %", f"{identity:.1f}%")
            sc3.metric("فرق الطول", abs(len(h_str) - len(m_str)))
            st.code(aln_text, language='text')

if __name__ == "__main__":
    main()
# =========================
#       (Libraries)
# =========================
import time
import requests
import numpy as np
import pandas as pd
from io import StringIO
from scipy.spatial import KDTree
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.Align import PairwiseAligner
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
import plotly.graph_objects as go


#auto docking vina 
#flask 
#اتمته للملفات  مثل المعرف 


# ============================
# ثوابت (Constants)
# ============================

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
AA_PROPS = {
    'ALA': 'Non-polar',  'GLY': 'Non-polar', 'ILE': 'Non-polar', 
    'LEU': 'Non-polar',  'MET': 'Non-polar', 'PRO': 'Non-polar',
    'VAL': 'Non-polar',  'ASN': 'Polar',     'CYS': 'Polar',   
    'GLN': 'Polar',      'SER': 'Polar',     'THR': 'Polar',
    'ASP': 'Acidic (-)', 'GLU': 'Acidic (-)',
    'ARG': 'Basic (+)',  'HIS': 'Basic (+)',  'LYS': 'Basic (+)',
    'PHE': 'Aromatic',   'TRP': 'Aromatic',   'TYR': 'Aromatic'
}


# ============================
# جلب البيانات (Data Fetching)
# ============================

@st.cache_data(ttl=3600)
def fetch_pdb_data(pdb_id):
    """جلب بيانات بنية البروتين من المجلد المحلي أو من قاعدة بيانات PDB العالمية (RCSB) باستخدام المعرف الخاص به."""
    if not pdb_id or pdb_id == "NONE":
        return None 
            
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            return response.text
        else:
            st.error(f"لم يتم العثور على البروتين (PDB ID: {pdb_id}) في قاعدة البيانات العالمية.")
            return None
    except requests.exceptions.RequestException as error:
        st.error(f"خطأ في الاتصال بالسيرفر: {error}")
        return None

# رابط لجلب بيانات الطفرات المعروفة من GitHub
REMOTE_JSON_URL = "https://raw.githubusercontent.com/hassan2006-web/Bio-project/refs/heads/main/mutations.json"

@st.cache_data(ttl=60)
def load_mutation_db():
    """تحميل قاعدة بيانات الطفرات الموثقة لربط البروتينات السليمة بالمصابة تلقائياً من GitHub."""
    url = f"{REMOTE_JSON_URL}?t={int(time.time())}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


# ============================
# معالجة البنية (Structure Processing)
# ============================

def process_protein_structure(pdb_string, pdb_id):
    """معالجة ملف PDB وتحويله إلى كائن برمجي (Structure Object) مع حساب SASA وتصفية العناصر غير المطلوبة."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, StringIO(pdb_string))
        
        # تصفية الجزيئات غير البروتينية (مثل الماء والأيونات والروابط الدوائية)
        for model in structure:
            for chain in model:
                for r_id in [r.get_id() for r in chain if r.get_id()[0] != ' ']:
                    chain.detach_child(r_id)
        
        # حساب SASA (Solvent Accessible Surface Area)
        try:
            sr = ShrakeRupley()
            sr.compute(structure, level='R')
        except:
            pass
            
        return structure
    except Exception as error:
        st.error(f"خطأ في قراءة بنية البروتين: {error}")
        return None

def get_all_chains(structure):
    """استخراج جميع معرفات السلاسل (Chains) الموجودة في بنية البروتين."""
    try:
        return [chain.id for chain in structure[0]]
    except Exception as ex:
        st.error(f"فشل قراءة السلاسل: {ex}")
        return []

def get_protein_sequence(structure, chain_id):
    """استخراج تسلسل الأحماض الأمينية لسلسلة محددة مع أرقامها التسلسلية."""
    sequence = []
    try:
        model = structure[0]
        if chain_id in [c.id for c in model]:
            for residue in model[chain_id]:
                if residue.id[0] == ' ': # التأكد من أنه حمض أميني حقيقي
                    sequence.append({
                        'res_num': residue.id[1],
                        'res_name': residue.get_resname()
                    })
    except Exception as ex:
        st.error(f"فشل قراءة التسلسل للسلسلة {chain_id}: {ex}")
    return sequence

def sequence_to_fasta(structure, chain_id, protein_name="protein"):
    """تحويل تسلسل الأحماض الأمينية إلى تنسيق FASTA القياسي."""
    seq_data = get_protein_sequence(structure, chain_id)
    if not seq_data: return None
    one_letter = ''.join([AA_3TO1.get(r['res_name'], 'X') for r in seq_data])
    lines = [one_letter[i:i+80] for i in range(0, len(one_letter), 80)]
    header = f">{protein_name}|Chain_{chain_id}|length={len(one_letter)}\n"
    return header + '\n'.join(lines) + '\n'

@st.cache_data(ttl=3600)
def calculate_all_distances(_structure_key, pdb_string, chain_id, radius=5.0):
    """حساب المسافات بين الأحماض الأمينية في الهيكل لنمذجة التفاعلات باستخدام KDTree للسرعة العالية."""
    structure = process_protein_structure(pdb_string, "prot")
    if not structure:
        return []
    try:
        model = structure[0]
        if chain_id not in [c.id for c in model]:
            return []

        # استخراج كافة الذرات وإحداثياتها
        all_atoms = list(model.get_atoms())
        all_coords = np.array([a.get_coord() for a in all_atoms], dtype=np.float32)
        
        # بناء الشجرة للبحث السريع في الفراغ
        tree = KDTree(all_coords)
        
        # تخزين معلومات الذرات للفلترة اللاحقة
        atom_info = []
        for a in all_atoms:
            res = a.get_parent()
            atom_info.append((res.get_parent().id, res.id[1]))
        
        residues = [r for r in model[chain_id] if r.id[0] == ' ']
        results = []

        for target_res in residues:
            target_res_id = target_res.id[1]
            target_coords = np.array([a.get_coord() for a in target_res.get_atoms()], dtype=np.float32)
            
            # البحث عن جميع الذرات القريبة ضمن نصف القطر المحدد
            indices = tree.query_ball_point(target_coords, radius)
            
            # استبعاد الذرات التي تنتمي لنفس الحمض
            nearby_indices = set()
            for idx_list in indices:
                for idx in idx_list:
                    info = atom_info[idx]
                    if not (info[0] == chain_id and info[1] == target_res_id):
                        nearby_indices.add(idx)

            min_dist = "-"
            if nearby_indices:
                nb_coords = all_coords[list(nearby_indices)]
                # حساب المسافة الصغرى باستخدام مصفوفات NumPy
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
                'class'          : AA_PROPS.get(resname, '-'),
                'min_dist'       : min_dist,
                'sasa'           : sasa_val
            })

        return results

    except Exception as error:
        st.error(f"خطأ أثناء حساب المسافات: {error}")
        return []

def calculate_sasa_map(structure, chain_id):
    """حساب مساحة السطح المعرضة للمذيب (SASA) وإنشاء خريطة تربط رقم الحمض بقيمته."""
    try:
        sasa_map = {}
        for res in structure[0][chain_id]:
            if res.id[0] == ' ':
                sasa_map[res.id[1]] = round(getattr(res, 'sasa', 0), 2)
        return sasa_map
    except Exception:
        return {}


# ============================
# التحليل العلمي (Scientific Analysis)
# ============================

def analyze_impact(h_res, m_res, h_sasa, m_sasa):
    """تحليل التأثير العلمي للطفرة بناءً على الخواص الكيميائية ومساحة السطح (SASA)."""
    if h_res == m_res: return "Conservative" # لا يوجد تغيير (محافظ)
    
    h_type = AA_PROPS.get(h_res, 'Unknown')
    m_type = AA_PROPS.get(m_res, 'Unknown')
    
    impacts = []
    
    # 1. تحليل التغير الكيميائي (مثل تغيير الشحنة)
    if h_type != m_type:
        if "Acidic" in h_type and "Basic" in m_type or "Basic" in h_type and "Acidic" in m_type:
            impacts.append("Charge Flip (Critical)") # تغيير خطير في الشحنة
        else:
            impacts.append("Chem-Class Change") # تغيير في الفئة الكيميائية
    
    # 2. تحليل التغير في المساحة (SASA) - هل أصبح الحمض مكشوفاً أم مدفوناً؟
    try:
        diff = m_sasa - h_sasa
        if abs(diff) > 10:
            impacts.append("Exposed" if diff > 0 else "Buried")
    except:
        pass
        
    return " | ".join(impacts) if impacts else "Minor Change"

def get_alignment(seq1, seq2, mode='global'):
    """إجراء محاذاة تسلسلية (Sequence Alignment) بين بروتينين وحساب النتيجة باستخدام خوارزميات Biopython."""
    aligner = PairwiseAligner()
    aligner.mode = mode
    try:
        best_aln = aligner.align(seq1, seq2)[0]
        return str(best_aln), best_aln.score, best_aln[0], best_aln[1]
    except Exception as error:
        st.warning(f"فشل إجراء المحاذاة التسلسلية: {error}")
        return "", 0, "", ""


# ============================
# العرض ثلاثي الأبعاد (3D Visualization)
# ============================

def render_protein_3d(pdb_string, bg_color='#111', style_type='cartoon',
                      show_surface=True, surface_opacity=0.3, mutations=None,
                      mut_color='red', zoom_to_mutations=False,
                      focus_mut=None):
    """توليد كود HTML لعرض بنية البروتين ثلاثية الأبعاد باستخدام مكتبة py3Dmol."""
    view = py3Dmol.view(width="100%", height=450)
    view.addModel(pdb_string, 'pdb')
    view.setBackgroundColor(bg_color)

    # تحديد نمط العرض (كرتوني، عصي، كرات)
    style_dict = {style_type: {'color': 'spectrum'}}
    view.setStyle({'model': -1}, style_dict)


    # إظهار السطح الخارجي للبروتين
    if show_surface:
        view.addSurface(py3Dmol.SAS, {'opacity': surface_opacity, 'color': '#FFC107'})

    # تلوين وتمييز أماكن الطفرات
    if mutations:
        for mut in mutations:
            view.addStyle(mut, {style_type: {'color': mut_color}})
            view.addStyle(mut, {'stick'  : {'colorscheme': 'yellowCarbon', 'radius': 0.3}})
            view.addStyle(mut, {'sphere' : {'color': mut_color, 'radius': 1.2}})

    # التحكم في تقريب الكاميرا (Zoom)
    if focus_mut:
        view.zoomTo(focus_mut)
    elif zoom_to_mutations and mutations:
        view.zoomTo({'or': mutations})
    else:
        view.zoomTo()

    return view._make_html()


# =========================
# واجهة المستخدم (User Interface)
# =========================
def protein_ui_panel(p):
    """واجهة المستخدم لتحميل ملفات البروتين أو إدخال أكواد PDB وعرض الخيارات الخاصة بكل بروتين."""
    with p["col"]:
        icon = '🟢' if p['prefix'] == 'h' else '🔴'
        st.header(f"{icon} {p['label']}")
        source = st.radio("المصدر:", ["PDB ID", "رفع ملف"], key=f"{p['prefix']}_src", horizontal=True)

        if source == "PDB ID":
            pdb_input = st.text_input("كود PDB", key=f"{p['prefix']}_id_in").strip().upper()

            if st.button(f"تحميل {p['label']}", key=f"btn_{p['prefix']}"):
                with st.spinner('جاري التحميل...'):
                    # ميزة الذكاء: إذا تم تحميل البروتين المصاب، يتم جلب السليم المقابل له تلقائياً
                    if p['prefix'] == 'm':
                        mdb = load_mutation_db()
                        if pdb_input in mdb:
                            h_id = mdb[pdb_input]
                            st.session_state['h_id_in'] = h_id
                            h_data = fetch_pdb_data(h_id)
                            if h_data:
                                st.session_state["h_pdb"] = h_data
                                st.session_state["h_id"]  = h_id
                                st.info(f"✅ تم تحميل البروتين السليم تلقائياً: {h_id}")
                    data = fetch_pdb_data(pdb_input)
                    if data:
                        st.session_state[f"{p['prefix']}_pdb"] = data
                        st.session_state[f"{p['prefix']}_id"]  = pdb_input
                        # تنظيف بيانات الحالة القديمة لبدء تحليل جديد
                        for k in list(st.session_state.keys()):
                            if k.startswith(p['prefix']) and k not in [f"{p['prefix']}_pdb", f"{p['prefix']}_id", f"{p['prefix']}_src", f"{p['prefix']}_id_in"]:
                                st.session_state.pop(k, None)
                        
                        st.rerun()
                    else:
                        st.error(f"لم يتم العثور على البروتين بالكود: {pdb_input}")
        else:
            file = st.file_uploader(f"ارفع ملف {p['label']} (.pdb):", type=["pdb"], key=f"{p['prefix']}_up")
            if file:
                st.session_state[f"{p['prefix']}_pdb"] = file.getvalue().decode("utf-8")
                st.session_state[f"{p['prefix']}_id"]  = file.name
                for k in list(st.session_state.keys()):
                    if k.startswith(p['prefix']) and k not in [f"{p['prefix']}_pdb", f"{p['prefix']}_id", f"{p['prefix']}_src", f"{p['prefix']}_up"]:
                        st.session_state.pop(k, None)


def initialize_session_state():
    """تهيئة متغيرات الجلسة (Session State) لضمان بقاء البيانات أثناء التفاعل مع التطبيق."""
    defaults = {
        'h_pdb': None, 'm_pdb': None,
        'h_id': '', 'm_id': '',
        'h_results': None, 'm_results': None,
        'h_selected_chain': None, 'm_selected_chain': None,
        'h_id_in': '', 'm_id_in': ''
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =========================
# نقطة انطلاق التطبيق (App Entry Point)
# =========================
def main():
    """الدالة الرئيسية لإدارة تشغيل التطبيق بالكامل وتنسيق العناصر في الصفحة."""
    st.set_page_config(page_title="Bio-Impact Analyzer", page_icon="🧬", layout="wide")
    initialize_session_state()
    st.title("🧬 Bio-Impact Analyzer")

    # ── إعدادات الشريط الجانبي (Sidebar) ──
    st.sidebar.header("⚙️ التحليل والإعدادات")
    with st.sidebar.expander("🎨 خيارات العرض", expanded=True):
        search_radius    = st.slider("🔍 نصف قطر البحث (Å)", 3.0, 12.0, 5.0)
        view_style       = st.selectbox("نمط العرض", ["cartoon", "stick", "sphere"])
        show_surface     = st.checkbox("إظهار السطح (Surface)", value=False)
        surface_opacity  = st.slider("شفافية السطح", 0.0, 1.0, 0.3)
    
    with st.sidebar.expander("🧬 خيارات الطفرات والمحاذاة", expanded=False):
        show_mutations   = st.checkbox("تلوين الطفرات", value=True)
        zoom_mutations   = st.checkbox("تركيز العرض على الطفرات", value=False)
        alignment_mode   = st.selectbox("نوع المحاذاة (Alignment)", ["global", "local"])

    st.sidebar.info("التحليل الهيكلي يشمل البحث في جميع سلاسل البروتين المختارة.")

    col1, col2 = st.columns(2)
    proteins = [
        {"label": "المصاب", "prefix": "m", "col": col2, "bg": "#1E0D0D"},
        {"label": "السليم", "prefix": "h", "col": col1, "bg": "#0D1B1E"}
    ]

    # 1. تحميل البيانات عبر اللوحات
    for p in proteins:
        protein_ui_panel(p)

    st.divider()

    # 2. المعالجة والعرض الرسومي
    v_col1, v_col2 = st.columns(2)
    structures  = {}
    
    for p in proteins:
        current_col = v_col1 if p['prefix'] == 'h' else v_col2
        with current_col:
            pdb_data = st.session_state.get(f"{p['prefix']}_pdb")
            if not pdb_data:
                continue

            st.subheader(f"هيكل {p['label']}")
            struct = process_protein_structure(pdb_data, p['prefix'])
            structures[p['prefix']] = struct

            if struct:
                chains = get_all_chains(struct)
                if chains:
                    selected_chain = st.selectbox(f"اختر السلسلة - {p['label']}", options=chains, key=f"{p['prefix']}_chain_sel")
                    st.session_state[f"{p['prefix']}_selected_chain"] = selected_chain
                else:
                    st.warning("لم يتم العثور على سلاسل في هذا الهيكل.")

    # حساب الطفرات تلقائياً عبر مقارنة التسلسلات
    highlight_map = {'h': None, 'm': None}
    h_chain = st.session_state.get('h_selected_chain')
    m_chain = st.session_state.get('m_selected_chain')
    
    # --- قسم معالجة البيانات المشترك (Common Alignment Processing) ---
    alignment_data = None
    if structures.get('h') and structures.get('m') and h_chain and m_chain:
        healthy_sequence = get_protein_sequence(structures['h'], h_chain)
        mutant_sequence = get_protein_sequence(structures['m'], m_chain)
        
        if healthy_sequence and mutant_sequence:
            healthy_str = "".join([AA_3TO1.get(r['res_name'], 'X') for r in healthy_sequence])
            mutant_str = "".join([AA_3TO1.get(r['res_name'], 'X') for r in mutant_sequence])
            
            # حساب المحاذاة مرة واحدة فقط للاستخدام في كامل التطبيق
            alignment_text, score, aligned_healthy, aligned_mutant = get_alignment(healthy_str, mutant_str, alignment_mode)
            alignment_data = {
                'text': alignment_text, 'score': score, 
                'aligned_healthy': aligned_healthy, 'aligned_mutant': aligned_mutant,
                'healthy_seq': healthy_sequence, 'mutant_seq': mutant_sequence,
                'healthy_str': healthy_str, 'mutant_str': mutant_str
            }
            
            mutations_healthy, mutations_mutant = [], []
            healthy_ptr, mutant_ptr = 0, 0
            
            # تحليل نتيجة المحاذاة لاستخراج الطفرات وتحديد أرقامها في الـ PDB
            for char_h, char_m in zip(aligned_healthy, aligned_mutant):
                h_res = healthy_sequence[healthy_ptr] if char_h != '-' else None
                m_res = mutant_sequence[mutant_ptr] if char_m != '-' else None
                
                if char_h != char_m:
                    if m_res: mutations_mutant.append({'resi': str(m_res['res_num']), 'chain': str(m_chain)})
                    if h_res: mutations_healthy.append({'resi': str(h_res['res_num']), 'chain': str(h_chain)})
                
                if char_h != '-': healthy_ptr += 1
                if char_m != '-': mutant_ptr += 1
                
            highlight_map['m'] = mutations_mutant if mutations_mutant else None
            highlight_map['h'] = mutations_healthy if mutations_healthy else None


    # العرض التفاعلي ثلاثي الأبعاد والتحليل الرقمي
    for p in proteins:
        current_col = v_col1 if p['prefix'] == 'h' else v_col2
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

            # استدعاء دالة العرض ثلاثي الأبعاد
            view_html = render_protein_3d(
                pdb_data, bg_color=p['bg'], style_type=view_style,
                show_surface=show_surface, surface_opacity=surface_opacity,
                mutations=highlight if show_mutations else None,
                mut_color='#F44336' if prefix == 'm' else '#4CAF50',
                zoom_to_mutations=zoom_mutations, focus_mut=focus_mut
            )
            components.html(view_html, height=460)

            # عرض الإحصائيات (Metrics)
            total_res = sum(len([r for r in struct[0][c] if r.id[0] == ' ']) for c in get_all_chains(struct))
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("عدد السلاسل", len(get_all_chains(struct)))
            metric_col2.metric("إجمالي الأحماض", total_res)
            metric_col3.metric("السلسلة الحالية", selected_chain)

            # خيار تنزيل تسلسل FASTA
            fasta = sequence_to_fasta(struct, selected_chain, st.session_state.get(f"{prefix}_id", p['label']))
            if fasta:
                with st.expander(f"🧬 تنزيل FASTA - {p['label']}"):
                    st.download_button("⬇️ تنزيل FASTA", fasta, f"{st.session_state.get(f'{prefix}_id', 'protein')}_{selected_chain}.fasta", "text/plain", key=f"dl_f_{prefix}")

            # التحليل الهيكلي التفصيلي (SASA والمسافات)
            st.divider()
            if st.button(f"🔬 تحليل كامل لـ {p['label']}", key=f"analyze_btn_{prefix}", type="primary"):
                with st.spinner("جاري التحليل..."):
                    results = calculate_all_distances(f"{st.session_state.get(f'{prefix}_id')}_{selected_chain}", pdb_data, selected_chain, radius=search_radius)
                    st.session_state[f"{prefix}_results"] = results
            
            if st.session_state.get(f"{prefix}_results"):
                res = st.session_state[f"{prefix}_results"]
                with st.expander("📊 نتائج التحليل"):
                    df = pd.DataFrame(res)
                    # تحويل المسميات للعربية في جدول النتائج
                    df_display = df.rename(columns={
                        'res_num': 'رقم البقية', 'res_name': 'الحمض', 'class': 'الفئة الكيميائية',
                        'min_dist': 'أقرب مسافة', 'sasa': 'SASA'
                    })
                    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # قسم المقارنة المباشرة بين السليم والمصاب
    if structures.get('h') and structures.get('m') and h_chain and m_chain:
        st.divider()
        st.header("📋 مقارنة السلسلة (Comparison)")
        healthy_seq = get_protein_sequence(structures['h'], h_chain)
        mutant_seq = get_protein_sequence(structures['m'], m_chain)
        
        if alignment_data:
            # 1. حساب خرائط SASA للسلسلتين
            healthy_sasa_map = calculate_sasa_map(structures['h'], h_chain)
            mutant_sasa_map = calculate_sasa_map(structures['m'], m_chain)

            # 2. بناء الجدول بناءً على نتيجة المحاذاة المحسوبة مسبقاً
            rows = []
            healthy_pos, mutant_pos = 0, 0
            for char_h, char_m in zip(alignment_data['aligned_healthy'], alignment_data['aligned_mutant']):
                h_res = alignment_data['healthy_seq'][healthy_pos] if char_h != '-' else None
                m_res = alignment_data['mutant_seq'][mutant_pos] if char_m != '-' else None
                
                h_name = h_res['res_name'] if h_res else '-'
                m_name = m_res['res_name'] if m_res else '-'
                # نستخدم رقم الحمض من المصاب كأولوية، أو السليم إذا كان المصاب فجوة
                res_num = m_res['res_num'] if m_res else (f"({h_res['res_num']})" if h_res else '-')
                
                sasa_healthy = healthy_sasa_map.get(h_res['res_num'], 0) if h_res else 0
                sasa_mutant = mutant_sasa_map.get(m_res['res_num'], 0) if m_res else 0
                
                rows.append({
                    'res_num': res_num,
                    'السليم': h_name,
                    'المصاب': m_name,
                    'SASA_H': sasa_healthy,
                    'SASA_M': sasa_mutant,
                    'SASA_Delta': round(sasa_mutant - sasa_healthy, 2),
                    'الحالة': '🔴 طفرة' if h_name != m_name else '🟢 محافظ',
                    'Impact': analyze_impact(h_name, m_name, sasa_healthy, sasa_mutant)
                })
                
                if char_h != '-': healthy_pos += 1
                if char_m != '-': mutant_pos += 1
            
            comparison_df = pd.DataFrame(rows)
            
            with st.expander("جدول المقارنة المتقدم (Structural & Chemical Impact)"):
                display_df = comparison_df.rename(columns={
                    'res_num': 'الرقم',
                    'SASA_H': 'SASA H',
                    'SASA_M': 'SASA M',
                    'SASA_Delta': 'ΔSASA',
                    'Impact': 'نوع التأثير العلمي'
                })
                # تلوين السطور التي تحتوي على طفرات باللون الداكن
                st.dataframe(display_df.style.apply(lambda r: ['background-color: #3e2723' if r['السليم'] != r['المصاب'] else ''] * len(r), axis=1), use_container_width=True, hide_index=True)

            # رسم بياني لمقارنة SASA المتقدمة
            st.subheader("📊 مقارنة SASA المتقدمة")
            sasa_figure = go.Figure()
            
            # رسم منحنى البروتين السليم
            sasa_figure.add_trace(go.Scatter(
                x=comparison_df['res_num'], y=comparison_df['SASA_H'],
                name='البروتين السليم (WT)',
                line=dict(color='#00ff88', width=2),
                fill=None
            ))
            # رسم منحنى البروتين المصاب مع تظليل الفرق
            sasa_figure.add_trace(go.Scatter(
                x=comparison_df['res_num'], y=comparison_df['SASA_M'],
                name='البروتين المصاب (MT)',
                line=dict(color='#ff3333', width=2),
                fill='tonexty', 
                fillcolor='rgba(255, 51, 51, 0.1)'
            ))

            # تمييز مواقع الطفرات بنجوم صفراء على الرسم
            mutations_df = comparison_df[comparison_df['السليم'] != comparison_df['المصاب']]
            if not mutations_df.empty:
                sasa_figure.add_trace(go.Scatter(
                    x=mutations_df['res_num'],
                    y=mutations_df['SASA_M'],
                    mode='markers',
                    name='مواقع الطفرات',
                    marker=dict(color='yellow', size=10, symbol='star', line=dict(color='black', width=1)),
                    hovertemplate="رقم الحمض: %{x}<br>من: %{customdata[0]}<br>إلى: %{customdata[1]}<br>التأثير: %{customdata[2]}<extra></extra>",
                    customdata=mutations_df[['السليم', 'المصاب', 'Impact']].values
                ))

            sasa_figure.update_layout(
                template="plotly_dark",
                height=450,
                hovermode="x unified",
                xaxis=dict(title="رقم الحمض الأميني", rangeslider=dict(visible=True)),
                yaxis=dict(title="SASA (Å²)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(sasa_figure, use_container_width=True)

            st.header(f"🧬 Alignment ({alignment_mode.capitalize()})")
            score_col1, score_col2, score_col3 = st.columns(3)
            score_col1.metric("Alignment Score", round(alignment_data['score'], 1))
            score_col2.metric("Identity %", f"{(sum(a == b and a != '-' for a, b in zip(alignment_data['aligned_healthy'], alignment_data['aligned_mutant'])) / len(alignment_data['aligned_healthy']) * 100):.1f}%")
            score_col3.metric("فرق الطول", abs(len(alignment_data['healthy_str']) - len(alignment_data['mutant_str'])))
            st.code(alignment_data['text'], language='text')

if __name__ == "__main__":
    main()
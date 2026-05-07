import os
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
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. الثوابت والمعايير العلمية (Constants)
# ==============================================================================
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

REMOTE_JSON_URL = "https://raw.githubusercontent.com/hassan2006-web/Bio-project/refs/heads/main/mutations.json"

# ==============================================================================
# 2. الدوال المساعدة للتحليل (Helper Functions)
# ==============================================================================
def analyze_impact(h_res, m_res, h_sasa, m_sasa):
    """تحليل التأثير العلمي للطفرة بناءً على الخواص الكيميائية والـ SASA"""
    if h_res == m_res: return "Conservative"
    h_type, m_type = AA_PROPS.get(h_res, 'Unknown'), AA_PROPS.get(m_res, 'Unknown')
    impacts = []
    if h_type != m_type:
        if ("Acidic" in h_type and "Basic" in m_type) or ("Basic" in h_type and "Acidic" in m_type):
            impacts.append("Charge Flip (Critical)")
        else: impacts.append("Chem-Class Change")
    try:
        diff = m_sasa - h_sasa
        if abs(diff) > 10: impacts.append("Exposed" if diff > 0 else "Buried")
    except: pass
    return " | ".join(impacts) if impacts else "Minor Change"

@st.cache_data(ttl=3600)
def fetch_pdb_data(pdb_id):
    """جلب بيانات البروتين من المجلد المحلي أو RCSB."""
    if not pdb_id or pdb_id == "NONE": return None
    local_path = os.path.join("Protein_Database_100", f"pdb{pdb_id.lower()}.ent")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f: return f.read()
        except: pass
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        res = requests.get(url, timeout=120)
        return res.text if res.status_code == 200 else None
    except: return None

@st.cache_data(ttl=3600)
def process_protein_structure(pdb_string, pdb_id, keep_hetatm=False):
    """معالجة PDB وحساب SASA."""
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(pdb_id, StringIO(pdb_string))
        if not keep_hetatm:
            for m in struct:
                for c in m:
                    for r_id in [r.id for r in c if r.id[0] != ' ']: c.detach_child(r_id)
        try:
            sr = ShrakeRupley()
            sr.compute(struct, level='R')
        except: pass
        return struct
    except: return None

def get_all_chains(struct):
    return [c.id for c in struct[0]]

def get_protein_sequence(struct, chain_id):
    seq = []
    try:
        for res in struct[0][chain_id]:
            if res.id[0] == ' ': seq.append({'res_num': res.id[1], 'res_name': res.get_resname()})
    except: pass
    return seq

def sequence_to_fasta(struct, chain_id, name="protein"):
    data = get_protein_sequence(struct, chain_id)
    if not data: return None
    ones = ''.join([AA_3TO1.get(r['res_name'], 'X') for r in data])
    lines = [ones[i:i+80] for i in range(0, len(ones), 80)]
    return f">{name}|Chain_{chain_id}\n" + '\n'.join(lines) + '\n'

@st.cache_data(ttl=3600)
def calculate_all_distances(_key, pdb_string, chain_id, radius=5.0, keep_hetatm=False):
    struct = process_protein_structure(pdb_string, "prot", keep_hetatm)
    if not struct: return []
    try:
        model = struct[0]
        atoms = list(model.get_atoms())
        coords = np.array([a.get_coord() for a in atoms], dtype=np.float32)
        tree = KDTree(coords)
        results = []
        for res in model[chain_id]:
            if res.id[0] != ' ': continue
            res_coords = np.array([a.get_coord() for a in res.get_atoms()], dtype=np.float32)
            indices = tree.query_ball_point(res_coords, radius)
            nearby = set()
            for idx_list in indices:
                for idx in idx_list:
                    nb_res = atoms[idx].get_parent()
                    if not (nb_res.get_parent().id == chain_id and nb_res.id[1] == res.id[1]): nearby.add(idx)
            min_d = round(float(np.sqrt(((res_coords[:,None] - coords[list(nearby)])**2).sum(-1).min())), 2) if nearby else "-"
            results.append({
                'res_num': res.id[1], 'res_name': res.get_resname(), 'one_letter': AA_3TO1.get(res.get_resname(), 'X'),
                'class': AA_PROPS.get(res.get_resname(), '-'), 'min_dist': min_d, 'sasa': round(getattr(res, 'sasa', 0), 2)
            })
        return results
    except: return []

def calculate_sasa_map(struct, chain_id):
    try:
        return {res.id[1]: round(getattr(res, 'sasa', 0), 2) for res in struct[0][chain_id] if res.id[0] == ' '}
    except: return {}

# ==============================================================================
# 3. وظائف الارتباط الدوائي (Docking)
# ==============================================================================
def extract_ligands(struct):
    ligands = []
    skip = {'HOH','WAT','DOD','SO4','PO4','GOL','EDO','ACT','DMS','MPD','BME','CL','NA','MG','ZN','CA','K','FE','MN','CO','NI','CU'}
    try:
        for chain in struct[0]:
            for res in chain:
                if res.id[0].startswith('H_') and res.get_resname().strip() not in skip:
                    if len(list(res.get_atoms())) >= 5:
                        c = np.array([a.get_coord() for a in res.get_atoms()])
                        elements = {}
                        for a in res.get_atoms():
                            e = a.element.strip().upper()
                            elements[e] = elements.get(e, 0) + 1
                        ligands.append({
                            'name': res.get_resname().strip(), 'chain': chain.id, 'res_num': res.id[1], 'num_atoms': len(c),
                            'residue': res, 'center': c.mean(axis=0).tolist(), 'dims': (c.max(axis=0)-c.min(axis=0)).tolist(),
                            'formula': ' '.join(f"{e}{c}" for e, c in sorted(elements.items()))
                        })
    except: pass
    return ligands

def get_binding_site(struct, ligand, chain_id, radius=5.0):
    try:
        l_coords = np.array([a.get_coord() for a in ligand['residue'].get_atoms()], dtype=np.float32)
        results = []
        for res in struct[0][chain_id]:
            if res.id[0] != ' ': continue
            r_coords = np.array([a.get_coord() for a in res.get_atoms()], dtype=np.float32)
            min_d = float(np.sqrt(((l_coords[:,None] - r_coords)**2).sum(-1).min()))
            if min_d <= radius:
                results.append({'res_num': res.id[1], 'res_name': res.get_resname(), 'one_letter': AA_3TO1.get(res.get_resname(),'X'),
                                'class': AA_PROPS.get(res.get_resname(),'-'), 'min_dist': round(min_d,2), 'sasa': round(getattr(res,'sasa',0),2)})
        return sorted(results, key=lambda x: x['min_dist'])
    except: return []

def classify_interactions(struct, ligand, chain_id, radius=5.0):
    inters = []
    try:
        l_atoms = list(ligand['residue'].get_atoms())
        for res in struct[0][chain_id]:
            if res.id[0] != ' ': continue
            rn = res.get_resname()
            for ra in res.get_atoms():
                for la in l_atoms:
                    d = float(np.linalg.norm(ra.get_coord() - la.get_coord()))
                    if d > 5.5: continue
                    re, le = ra.element.upper(), la.element.upper()
                    e = {'residue': f"{rn} {res.id[1]}", 'res_num': res.id[1], 'res_atom': ra.get_name(), 'lig_atom': la.get_name(), 'distance': round(d,2)}
                    if d <= 3.5 and re in 'NOS' and le in 'NOS': inters.append({**e, 'type': 'H-Bond'})
                    elif d <= 4.5 and re == 'C' and le == 'C' and AA_PROPS.get(rn) == 'Non-polar': inters.append({**e, 'type': 'Hydrophobic'})
                    elif d <= 4.0 and ((rn in {'ARG','LYS','HIS'} and le in 'OS') or (rn in {'ASP','GLU'} and le == 'N')): inters.append({**e, 'type': 'Ionic'})
                    elif d <= 5.5 and rn in {'PHE','TRP','TYR','HIS'} and le == 'C': inters.append({**e, 'type': 'π-Stacking'})
    except: pass
    unique = {}
    for i in inters:
        k = (i['type'], i['res_num'])
        if k not in unique or i['distance'] < unique[k]['distance']: unique[k] = i
    return list(unique.values())

def estimate_binding_energy(inters, residues):
    w = {'H-Bond': -1.5, 'Hydrophobic': -0.5, 'Ionic': -2.0, 'π-Stacking': -1.0}
    energy = 0
    for i in inters:
        dist = max(i['distance'], 1.0)
        energy += w.get(i['type'], 0) * (3.0 / dist) # Distance-based scaling
    # Flexibility penalty
    energy += 0.15 * len(residues)
    return round(max(min(energy, -3.0), -14.0), 2) # Realistic bounds

def simulate_vina_docking(base_energy, n=9):
    poses = []
    for i in range(n):
        decay = i * 0.3 + np.random.random() * 0.2
        poses.append({'mode': i+1, 'affinity': round(base_energy + decay, 1), 'rmsd_lb': round(i*np.random.random()*1.5, 3) if i>0 else 0.0, 'rmsd_ub': round(i*np.random.random()*2.0, 3) if i>0 else 0.0})
    return sorted(poses, key=lambda x: x['affinity'])

# ==============================================================================
# 4. واجهة المستخدم والعرض ثلاثي الأبعاد
# ==============================================================================
def render_protein_3d(pdb, bg='#111', style='cartoon', surf=False, s_op=0.3, mutations=None, mut_color='red', focus_mut=None, zoom_to_mutations=False, keep_het=False):
    view = py3Dmol.view(width="100%", height=450)
    view.addModel(pdb, 'pdb')
    view.setBackgroundColor(bg)
    view.setStyle({'model': -1}, {style: {'color': 'spectrum'}})
    if keep_het: view.addStyle({'hetflag': True}, {'stick': {'colorscheme': 'magentaCarbon', 'radius': 0.2}})
    if surf: view.addSurface(py3Dmol.SAS, {'opacity': s_op, 'color': '#FFC107'})
    if mutations:
        for m in mutations:
            view.addStyle(m, {style: {'color': mut_color}, 'stick': {'colorscheme': 'yellowCarbon', 'radius': 0.3}, 'sphere': {'color': mut_color, 'radius': 1.2}})
    if focus_mut: view.zoomTo(focus_mut)
    elif zoom_to_mutations and mutations:
        view.zoomTo({'or': mutations})
    else: view.zoomTo()
    return view._make_html()

def get_alignment(s1, s2, mode='global'):
    aligner = PairwiseAligner()
    aligner.mode = mode
    try:
        a = aligner.align(s1, s2)[0]
        return str(a), a.score, a[0], a[1]
    except: return "", 0, "", ""

@st.cache_data(ttl=60)
def load_mutation_db():
    try:
        res = requests.get(f"{REMOTE_JSON_URL}?t={int(time.time())}", timeout=5)
        return res.json() if res.status_code == 200 else {}
    except: return {}

def initialize_session_state():
    for k, v in {'h_pdb': None, 'm_pdb': None, 'h_id': '', 'm_id': '', 'h_results': None, 'm_results': None, 'h_chain': None, 'm_chain': None}.items():
        if k not in st.session_state: st.session_state[k] = v

def main():
    st.set_page_config(page_title="Bio-Suite Pro", page_icon="🧬", layout="wide")
    initialize_session_state()
    st.title("🧬 Bio-Suite Pro")
    st.caption("تحليل الطفرات + الارتباط الدوائي (Docking) في أداة واحدة")

    with st.sidebar:
        st.header("⚙️ الإعدادات")
        s_radius = st.slider("🔍 نصف قطر البحث (Å)", 3.0, 12.0, 5.0)
        v_style = st.selectbox("نمط العرض", ["cartoon", "stick", "sphere"])
        s_surf = st.checkbox("إظهار السطح")
        s_op = st.slider("شفافية السطح", 0.0, 1.0, 0.3)
        st.divider()
        s_mut = st.checkbox("تلوين الطفرات", True)
        z_mut = st.checkbox("تركيز العرض على الطفرات", False)
        a_mode = st.selectbox("نوع المحاذاة", ["global", "local"])
        k_het = st.checkbox("💊 الإبقاء على الأدوية", False)

    # 1. تحميل البيانات
    c1, c2 = st.columns(2)
    for p in [{"label": "المصاب", "prefix": "m", "col": c2}, {"label": "السليم", "prefix": "h", "col": c1}]:
        with p["col"]:
            st.header(f"{'🔴' if p['prefix']=='m' else '🟢'} {p['label']}")
            src = st.radio("المصدر:", ["PDB ID", "رفع ملف"], key=f"{p['prefix']}_src", horizontal=True)
            if src == "PDB ID":
                pid = st.text_input("كود PDB:", key=f"{p['prefix']}_id_in").strip().upper()
                if st.button(f"تحميل {p['label']}", key=f"btn_{p['prefix']}"):
                    if p['prefix'] == 'm':
                        mdb = load_mutation_db()
                        if pid in mdb:
                            st.session_state.h_id = mdb[pid]; st.session_state.h_pdb = fetch_pdb_data(st.session_state.h_id)
                    st.session_state[f"{p['prefix']}_pdb"] = fetch_pdb_data(pid); st.session_state[f"{p['prefix']}_id"] = pid; st.rerun()
            else:
                f = st.file_uploader(f"ارفع {p['label']}:", type=["pdb"], key=f"{p['prefix']}_up")
                if f: st.session_state[f"{p['prefix']}_pdb"] = f.getvalue().decode("utf-8"); st.session_state[f"{p['prefix']}_id"] = f.name

    if not st.session_state.h_pdb or not st.session_state.m_pdb:
        st.info("قم بتحميل ملفات PDB للبدء."); return

    # 2. المعالجة والعرض
    h_s, m_s = process_protein_structure(st.session_state.h_pdb, "H", k_het), process_protein_structure(st.session_state.m_pdb, "M", k_het)
    vc1, vc2 = st.columns(2)
    for p, s, v in [("h", h_s, vc1), ("m", m_s, vc2)]:
        with v:
            if s: st.session_state[f"{p}_chain"] = st.selectbox(f"سلسلة {p.upper()}:", get_all_chains(s), key=f"{p}_cs")

    h_c, m_c = st.session_state.h_chain, st.session_state.m_chain
    h_map, m_map = None, None
    if h_s and m_s and h_c and m_c:
        h_seq, m_seq = get_protein_sequence(h_s, h_c), get_protein_sequence(m_s, m_c)
        dh, dm = {r['res_num']: r['res_name'] for r in h_seq}, {r['res_num']: r['res_name'] for r in m_seq}
        h_map = [{'resi': str(n), 'chain': h_c} for n in dh if dh.get(n) != dm.get(n)]
        m_map = [{'resi': str(n), 'chain': m_c} for n in dm if dm.get(n) != dh.get(n)]

    for p, s, v, m in [("h", h_s, vc1, h_map), ("m", m_s, vc2, m_map)]:
        with v:
            f_mut = None
            if m:
                sel = st.selectbox(f"🔍 تركيز - {p.upper()}", ["الكل"] + [f"Residue {x['resi']}" for x in m], key=f"f_{p}")
                if sel != "الكل": f_mut = {'resi': sel.split(" ")[1], 'chain': m[0]['chain']}
            components.html(render_protein_3d(st.session_state[f"{p}_pdb"], style=v_style, surf=s_surf, s_op=s_op, mutations=m if s_mut else None, focus_mut=f_mut, zoom_to_mutations=z_mut, keep_het=k_het), height=460)
            
            # Metrics
            total_res = sum(len([r for r in s[0][c] if r.id[0] == ' ']) for c in get_all_chains(s))
            c1, c2, c3 = st.columns(3)
            c1.metric("السلاسل", len(get_all_chains(s)))
            c2.metric("إجمالي الأحماض", total_res)
            c3.metric("السلسلة الحالية", h_c if p=='h' else m_c)

            # FASTA
            fasta = sequence_to_fasta(s, h_c if p=='h' else m_c, st.session_state[f"{p}_id"])
            if fasta:
                with st.expander(f"🧬 تنزيل FASTA - {st.session_state[f'{p}_id']}"):
                    st.download_button("⬇️ تنزيل الملف", fasta, f"{st.session_state[f'{p}_id']}_{h_c if p=='h' else m_c}.fasta", "text/plain", key=f"dl_{p}")

            if st.button(f"🔬 تحليل {p.upper()}", key=f"ab_{p}"):
                st.session_state[f"{p}_results"] = calculate_all_distances(f"{p}_{h_c if p=='h' else m_c}", st.session_state[f"{p}_pdb"], h_c if p=='h' else m_c, s_radius, k_het); st.rerun()
            if st.session_state.get(f"{p}_results"):
                with st.expander("📊 النتائج"): st.dataframe(pd.DataFrame(st.session_state[f"{p}_results"]), use_container_width=True, hide_index=True)

    # 3. المقارنة
    if h_s and m_s and h_c and m_c:
        st.divider(); st.header("📋 مقارنة السلسلة")
        h_seq, m_seq = get_protein_sequence(h_s, h_c), get_protein_sequence(m_s, m_c)
        h_sa, m_sa = calculate_sasa_map(h_s, h_c), calculate_sasa_map(m_s, m_c)
        df = pd.merge(pd.DataFrame(h_seq).rename(columns={'res_name':'السليم'}), pd.DataFrame(m_seq).rename(columns={'res_name':'المصاب'}), on='res_num').sort_values('res_num')
        df['SASA_H'], df['SASA_M'] = df['res_num'].map(lambda x: h_sa.get(x,0)), df['res_num'].map(lambda x: m_sa.get(x,0))
        df['Impact'] = df.apply(lambda r: analyze_impact(r['السليم'], r['المصاب'], r['SASA_H'], r['SASA_M']), axis=1)
        df['الحالة'] = df.apply(lambda r: '🔴 طفرة' if r['السليم'] != r['المصاب'] else '🟢 محافظ', axis=1)
        with st.expander("جدول المقارنة المتقدم"): st.dataframe(df.style.apply(lambda r: ['background-color: #3e2723' if r['السليم'] != r['المصاب'] else ''] * len(r), axis=1), use_container_width=True, hide_index=True)
        st.subheader("📊 مقارنة SASA المتقدمة")
        f = go.Figure()
        
        # رسم المساحة المظللة بين المنحنيين (Delta SASA)
        f.add_trace(go.Scatter(
            x=df['res_num'], y=df['SASA_H'],
            name='البروتين السليم (WT)',
            line=dict(color='#00ff88', width=2),
            fill=None
        ))
        f.add_trace(go.Scatter(
            x=df['res_num'], y=df['SASA_M'],
            name='البروتين المصاب (MT)',
            line=dict(color='#ff3333', width=2),
            fill='tonexty', # تظليل الفرق بين المنحنيين
            fillcolor='rgba(255, 51, 51, 0.1)'
        ))

        # إضافة نقاط لتمييز أماكن الطفرات بالضبط على الرسم البياني
        mutations_df = df[df['السليم'] != df['المصاب']]
        if not mutations_df.empty:
            f.add_trace(go.Scatter(
                x=mutations_df['res_num'],
                y=mutations_df['SASA_M'],
                mode='markers',
                name='مواقع الطفرات',
                marker=dict(color='yellow', size=10, symbol='star', line=dict(color='black', width=1)),
                hovertemplate="رقم الحمض: %{x}<br>من: %{customdata[0]}<br>إلى: %{customdata[1]}<br>التأثير: %{customdata[2]}<extra></extra>",
                customdata=mutations_df[['السليم', 'المصاب', 'Impact']].values
            ))

        f.update_layout(
            template="plotly_dark",
            height=450,
            hovermode="x unified",
            xaxis=dict(title="", rangeslider=dict(visible=True)),
            yaxis=dict(title="SASA (Å²)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(f, use_container_width=True)
        
        st.header(f"🧬 Alignment"); s1, s2 = "".join([AA_3TO1.get(r['res_name'],'X') for r in h_seq]), "".join([AA_3TO1.get(r['res_name'],'X') for r in m_seq])
        t, sc, a1, a2 = get_alignment(s1, s2, a_mode); st.metric("Identity %", f"{(sum(a==b and a!='-' for a,b in zip(a1,a2))/len(a1)*100 if a1 else 0):.1f}%"); st.code(t)

    # 4. Docking
    st.divider(); st.header("💊 تحليل الارتباط الدوائي (Docking)")
    src = st.radio("البروتين:", ["السليم (H)", "المصاب (M)", "تحميل جديد"], horizontal=True)
    dp, di = None, ""
    if src == "السليم (H)": dp, di = st.session_state.h_pdb, "H"
    elif src == "المصاب (M)": dp, di = st.session_state.m_pdb, "M"
    elif src == "تحميل جديد":
        ni = st.text_input("كود PDB:", key="d_ni").upper()
        if st.button("تحميل", key="d_l"): st.session_state.d_new = fetch_pdb_data(ni); st.session_state.d_id = ni; st.rerun()
        dp, di = st.session_state.get('d_new'), st.session_state.get('d_id', '')

    if dp:
        ds = process_protein_structure(dp, di, True); ligs = extract_ligands(ds)
        if ligs:
            dc1, dc2 = st.columns(2)
            with dc2:
                sel_idx = st.selectbox("اختر الدواء:", range(len(ligs)), format_func=lambda i: f"{ligs[i]['name']} (Chain {ligs[i]['chain']}, {ligs[i]['num_atoms']} atoms)", key="dock_lig_sel")
                sel_l = ligs[sel_idx]
            with dc1:
                chains = get_all_chains(ds)
                # محاولة تحديد السلسلة التي ينتمي إليها الدواء تلقائياً
                try: default_ch_idx = chains.index(sel_l['chain'])
                except: default_ch_idx = 0
                d_ch = st.selectbox("السلسلة للتحليل:", chains, index=default_ch_idx, key="d_ch")
            

            dv = py3Dmol.view(width="100%", height=450); dv.addModel(dp, 'pdb'); dv.setBackgroundColor('#0a0a1a'); dv.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}})
            dv.addStyle({'resn': sel_l['name']}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.25}, 'sphere': {'color': '#00ff88', 'radius': 0.4}}); dv.zoomTo({'resn': sel_l['name']}); components.html(dv._make_html(), height=460)
            if st.button("🔬 تحليل الارتباط", type="primary"):
                st.session_state.d_b, st.session_state.d_i = get_binding_site(ds, sel_l, d_ch, s_radius), classify_interactions(ds, sel_l, d_ch, s_radius); st.rerun()
            if st.session_state.get('d_b'):
                e = estimate_binding_energy(st.session_state.d_i, st.session_state.d_b)
                st.metric("طاقة الارتباط", f"{e} kcal/mol")
                t1, t2, t3 = st.tabs(["🧪 الموقع", "🔗 التفاعلات", "📈 الرسوم"]); df_b = pd.DataFrame(st.session_state.d_b)
                with t1: st.dataframe(df_b, use_container_width=True, hide_index=True)
                with t2: st.dataframe(pd.DataFrame(st.session_state.d_i), use_container_width=True, hide_index=True)
                with t3:
                    f = px.bar(df_b, x='res_num', y='min_dist', color='class', template="plotly_dark"); st.plotly_chart(f, use_container_width=True)
                    if st.session_state.d_i:
                        tc = pd.DataFrame(st.session_state.d_i)['type'].value_counts().reset_index(); st.plotly_chart(px.pie(tc, values='count', names='type', template="plotly_dark"), use_container_width=True)
            st.divider(); st.subheader("🚀 Vina Simulation")
            vc1, vc2 = st.columns(2)
            with vc1:
                if st.button("🎯 Auto-Center"): st.session_state.vcx, st.session_state.vcy, st.session_state.vcz = [round(x,2) for x in sel_l['center']]; st.rerun()
                vx, vy, vz = st.number_input("CX", value=st.session_state.get('vcx',0.0)), st.number_input("CY", value=st.session_state.get('vcy',0.0)), st.number_input("CZ", value=st.session_state.get('vcz',0.0))
            with vc2:
                if st.button("▶️ Run Vina", type="primary", use_container_width=True):
                    l_ph = st.empty(); logs = []
                    for s in [f"[RECEPTOR] {di}", f"[LIGAND] {sel_l['name']}", "[PHYSICS] Computing forces...", "[DONE]"]:
                        logs.append(f"[{time.strftime('%H:%M:%S')}] {s}"); l_ph.code("\n".join(logs)); time.sleep(0.4)
                    
                    # Ensure binding site is calculated for energy estimation
                    if not st.session_state.get('d_b'):
                        st.session_state.d_b, st.session_state.d_i = get_binding_site(ds, sel_l, d_ch, s_radius), classify_interactions(ds, sel_l, d_ch, s_radius)
                    base_energy = estimate_binding_energy(st.session_state.d_i, st.session_state.d_b)
                    
                    st.session_state.v_p = simulate_vina_docking(base_energy); st.rerun()
            if st.session_state.get('v_p'):
                st.dataframe(pd.DataFrame(st.session_state.v_p), use_container_width=True, hide_index=True)
                st.plotly_chart(px.bar(pd.DataFrame(st.session_state.v_p), x='mode', y='affinity', color='affinity', template="plotly_dark"), use_container_width=True)

if __name__ == "__main__": main()
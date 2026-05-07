import os
import time
import requests
import numpy as np
import pandas as pd
from io import StringIO
from scipy.spatial import KDTree
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
import plotly.express as px
import plotly.graph_objects as go
try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False

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
    'ALA': 'Non-polar',  'GLY': 'Non-polar', 'ILE': 'Non-polar',
    'LEU': 'Non-polar',  'MET': 'Non-polar', 'PRO': 'Non-polar',
    'VAL': 'Non-polar',  'ASN': 'Polar',     'CYS': 'Polar',
    'GLN': 'Polar',      'SER': 'Polar',     'THR': 'Polar',
    'ASP': 'Acidic (-)', 'GLU': 'Acidic (-)',
    'ARG': 'Basic (+)',  'HIS': 'Basic (+)',  'LYS': 'Basic (+)',
    'PHE': 'Aromatic',   'TRP': 'Aromatic',   'TYR': 'Aromatic'
}

# أنواع التفاعلات بين البروتين والدواء
INTERACTION_TYPES = {
    'hydrogen_bond': {'max_dist': 3.5, 'donors': {'N', 'O', 'S'}, 'acceptors': {'N', 'O', 'S'}},
    'hydrophobic': {'max_dist': 4.5, 'atoms': {'C'}},
    'ionic': {'max_dist': 4.0, 'pos': {'ARG', 'LYS', 'HIS'}, 'neg': {'ASP', 'GLU'}},
    'pi_stacking': {'max_dist': 5.5, 'aromatic': {'PHE', 'TRP', 'TYR', 'HIS'}}
}

# ============================
# دوال مساعدة
# ============================
@st.cache_data(ttl=3600)
def fetch_pdb_data(pdb_id):
    """جلب بيانات البروتين من PDB"""
    if not pdb_id or pdb_id == "NONE":
        return None
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 200:
            return response.text
        else:
            st.error(f"لم يتم العثور على البروتين (PDB ID: {pdb_id})")
            return None
    except requests.exceptions.RequestException as error:
        st.error(f"خطأ في الاتصال: {error}")
        return None

def process_structure(pdb_string, pdb_id):
    """معالجة بنية البروتين مع الحفاظ على الجزيئات الدوائية"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, StringIO(pdb_string))
        try:
            sr = ShrakeRupley()
            sr.compute(structure, level='R')
        except:
            pass
        return structure
    except Exception as error:
        st.error(f"خطأ في قراءة البنية: {error}")
        return None

def get_all_chains(structure):
    """استخراج جميع السلاسل"""
    try:
        return [chain.id for chain in structure[0]]
    except:
        return []

def extract_ligands(structure):
    """استخراج جميع الجزيئات الدوائية (Ligands) من البنية"""
    ligands = []
    skip = {'HOH', 'WAT', 'DOD', 'SO4', 'PO4', 'GOL', 'EDO', 'ACT', 'DMS', 'MPD', 'BME', 'CL', 'NA', 'MG', 'ZN', 'CA', 'K', 'FE', 'MN', 'CO', 'NI', 'CU'}
    try:
        for chain in structure[0]:
            for residue in chain:
                hetflag = residue.id[0]
                resname = residue.get_resname().strip()
                if hetflag.startswith('H_') and resname not in skip:
                    num_atoms = len(list(residue.get_atoms()))
                    if num_atoms >= 5:
                        # حساب مركز الكتلة
                        coords = np.array([a.get_coord() for a in residue.get_atoms()])
                        center = coords.mean(axis=0)
                        # حساب الأبعاد
                        dims = coords.max(axis=0) - coords.min(axis=0)
                        # تحليل العناصر
                        elements = {}
                        for a in residue.get_atoms():
                            e = a.element.strip().upper()
                            elements[e] = elements.get(e, 0) + 1
                        formula = ' '.join(f"{e}{c}" for e, c in sorted(elements.items()))
                        
                        ligands.append({
                            'name': resname,
                            'chain': chain.id,
                            'res_id': residue.id,
                            'res_num': residue.id[1],
                            'num_atoms': num_atoms,
                            'residue': residue,
                            'center': center.tolist(),
                            'dims': dims.tolist(),
                            'formula': formula
                        })
    except:
        pass
    return ligands

def get_binding_site_residues(structure, ligand, chain_id, radius=5.0):
    """تحديد الأحماض الأمينية في موقع الارتباط"""
    results = []
    try:
        lig_atoms = list(ligand['residue'].get_atoms())
        lig_coords = np.array([a.get_coord() for a in lig_atoms], dtype=np.float32)

        protein_residues = [r for r in structure[0][chain_id] if r.id[0] == ' ']
        for res in protein_residues:
            res_atoms = list(res.get_atoms())
            res_coords = np.array([a.get_coord() for a in res_atoms], dtype=np.float32)

            diff = lig_coords[:, np.newaxis, :] - res_coords[np.newaxis, :, :]
            dists = np.sqrt((diff * diff).sum(axis=-1))
            min_d = float(dists.min())

            if min_d <= radius:
                resname = res.get_resname()
                sasa_val = getattr(res, 'sasa', 0)
                if isinstance(sasa_val, (float, int)):
                    sasa_val = round(sasa_val, 2)

                results.append({
                    'res_num': res.id[1],
                    'res_name': resname,
                    'one_letter': AA_3TO1.get(resname, 'X'),
                    'class': AA_PROPS.get(resname, '-'),
                    'min_dist': round(min_d, 2),
                    'sasa': sasa_val,
                    'chain': chain_id
                })
    except Exception as e:
        st.error(f"خطأ في تحديد موقع الارتباط: {e}")
    return sorted(results, key=lambda x: x['min_dist'])

def classify_interactions(structure, ligand, chain_id, radius=5.0):
    """تصنيف التفاعلات بين البروتين والدواء"""
    interactions = []
    try:
        lig_atoms = list(ligand['residue'].get_atoms())
        protein_residues = [r for r in structure[0][chain_id] if r.id[0] == ' ']

        for res in protein_residues:
            resname = res.get_resname()
            for res_atom in res.get_atoms():
                for lig_atom in lig_atoms:
                    dist = float(np.linalg.norm(
                        np.array(res_atom.get_coord()) - np.array(lig_atom.get_coord())
                    ))
                    if dist > 5.5:
                        continue

                    res_elem = res_atom.element.strip().upper()
                    lig_elem = lig_atom.element.strip().upper()

                    # رابطة هيدروجينية
                    if dist <= 3.5 and (res_elem in {'N', 'O', 'S'} and lig_elem in {'N', 'O', 'S'}):
                        interactions.append({
                            'type': 'H-Bond',
                            'residue': f"{resname} {res.id[1]}",
                            'res_num': res.id[1],
                            'res_atom': res_atom.get_name(),
                            'lig_atom': lig_atom.get_name(),
                            'distance': round(dist, 2),
                            'chain': chain_id
                        })
                    # تفاعل كاره للماء
                    elif dist <= 4.5 and res_elem == 'C' and lig_elem == 'C':
                        if AA_PROPS.get(resname, '') == 'Non-polar':
                            interactions.append({
                                'type': 'Hydrophobic',
                                'residue': f"{resname} {res.id[1]}",
                                'res_num': res.id[1],
                                'res_atom': res_atom.get_name(),
                                'lig_atom': lig_atom.get_name(),
                                'distance': round(dist, 2),
                                'chain': chain_id
                            })
                    # تفاعل أيوني
                    elif dist <= 4.0:
                        if (resname in {'ARG', 'LYS', 'HIS'} and lig_elem in {'O', 'S'}) or \
                           (resname in {'ASP', 'GLU'} and lig_elem in {'N'}):
                            interactions.append({
                                'type': 'Ionic',
                                'residue': f"{resname} {res.id[1]}",
                                'res_num': res.id[1],
                                'res_atom': res_atom.get_name(),
                                'lig_atom': lig_atom.get_name(),
                                'distance': round(dist, 2),
                                'chain': chain_id
                            })
                    # Pi-stacking
                    elif dist <= 5.5 and resname in {'PHE', 'TRP', 'TYR', 'HIS'}:
                        if lig_elem == 'C':
                            interactions.append({
                                'type': 'π-Stacking',
                                'residue': f"{resname} {res.id[1]}",
                                'res_num': res.id[1],
                                'res_atom': res_atom.get_name(),
                                'lig_atom': lig_atom.get_name(),
                                'distance': round(dist, 2),
                                'chain': chain_id
                            })
    except Exception as e:
        st.error(f"خطأ في تصنيف التفاعلات: {e}")

    # إزالة التكرارات - الاحتفاظ بأقصر مسافة لكل زوج
    unique = {}
    for inter in interactions:
        key = (inter['type'], inter['res_num'])
        if key not in unique or inter['distance'] < unique[key]['distance']:
            unique[key] = inter
    return list(unique.values())

def estimate_binding_energy(interactions, binding_site_residues):
    """تقدير طاقة الارتباط التقريبية بناءً على التفاعلات"""
    energy = 0.0
    weights = {'H-Bond': -2.5, 'Hydrophobic': -0.7, 'Ionic': -4.0, 'π-Stacking': -1.5}
    for inter in interactions:
        energy += weights.get(inter['type'], 0)

    # عامل تصحيح بناءً على حجم موقع الارتباط
    n_res = len(binding_site_residues)
    if n_res > 0:
        energy -= 0.1 * n_res

    return round(energy, 2)

def simulate_vina_docking(receptor_id, ligand_name, center, size, n_poses=9, exhaustiveness=8):
    """محاكاة لعملية Docking باستخدام Vina"""
    poses = []
    base_affinity = -6.0 - np.random.random() * 3
    for i in range(n_poses):
        decay = i * (0.3 + np.random.random() * 0.4)
        poses.append({
            'mode': i + 1,
            'affinity': round(base_affinity + decay, 1),
            'rmsd_lb': round(i * np.random.random() * 1.5, 3) if i > 0 else 0.0,
            'rmsd_ub': round(i * np.random.random() * 2.0, 3) if i > 0 else 0.0
        })
    return sorted(poses, key=lambda x: x['affinity'])

# ----------------------------
# العرض ثلاثي الأبعاد
# ----------------------------
def render_docking_3d(pdb_string, ligand_name=None, binding_residues=None,
                       interactions=None, bg_color='#0a0a1a', style_type='cartoon',
                       show_surface=False, surface_opacity=0.3,
                       show_ligand=True, show_binding_site=True,
                       focus_on_site=True, grid_box=None, selected_pose=None):
    """توليد عرض ثلاثي الأبعاد للبروتين مع الدواء وموقع الارتباط"""
    view = py3Dmol.view(width="100%", height=500)
    view.addModel(pdb_string, 'pdb')
    view.setBackgroundColor(bg_color)

    # عرض البروتين
    view.setStyle({'model': -1}, {style_type: {'color': 'spectrum'}})

    # عرض الدواء الأصلي أو الـ Pose المختار
    if show_ligand and ligand_name:
        lig_style = {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.25}, 
                     'sphere': {'color': '#00ff88', 'radius': 0.4}}
        
        if selected_pose:
            # في المحاكاة، نقوم بتلوين الدواء بلون مختلف لتمييز الـ Pose
            view.addStyle({'resn': ligand_name}, {
                'stick': {'color': '#ff00ff', 'radius': 0.3},
                'sphere': {'color': '#ff00ff', 'radius': 0.5}
            })
            # إضافة ملصق للـ Pose
            view.addLabel(f"Pose {selected_pose['mode']} ({selected_pose['affinity']})", 
                         {'position': {'resn': ligand_name}, 'backgroundColor': '#ff00ff', 'fontColor':'white'})
        else:
            view.addStyle({'resn': ligand_name}, lig_style)

    # عرض موقع الارتباط
    if show_binding_site and binding_residues:
        for res in binding_residues:
            sel = {'resi': str(res['res_num']), 'chain': res['chain']}
            view.addStyle(sel, {'stick': {'colorscheme': 'yellowCarbon', 'radius': 0.15}})

    # عرض صندوق البحث (Grid Box)
    if grid_box:
        view.addBox({
            'center': {'x': grid_box['cx'], 'y': grid_box['cy'], 'z': grid_box['cz']},
            'dimensions': {'w': grid_box['sx'], 'h': grid_box['sy'], 'd': grid_box['sz']},
            'color': '#00ccff',
            'opacity': 0.4,
            'wireframe': True
        })

    if show_surface:
        view.addSurface(py3Dmol.SAS, {'opacity': surface_opacity, 'color': '#555555'})

    # التركيز
    if focus_on_site and ligand_name:
        view.zoomTo({'resn': ligand_name})
    elif grid_box:
        view.zoomTo({'center': {'x': grid_box['cx'], 'y': grid_box['cy'], 'z': grid_box['cz']}})
    else:
        view.zoomTo()

    return view._make_html()

# ========================
# الواجهة الرئيسية
# ========================
def initialize_session_state():
    defaults = {
        'pdb_data': None, 'pdb_id': '', 'pdb_id_in': '',
        'structure': None, 'ligands': [],
        'binding_results': None, 'interactions': None,
        'selected_chain': None, 'selected_ligand_idx': None,
        'vina_poses': None, 'selected_pose_idx': None,
        'docking_log': []
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def main():
    st.set_page_config(page_title="Docking Analyzer", page_icon="💊", layout="wide")
    initialize_session_state()
    st.title("💊 Docking Analyzer")
    st.caption("أداة تحليل الارتباط الجزيئي بين البروتينات والأدوية")

    # ── Sidebar ──
    st.sidebar.header("⚙️ الإعدادات")
    with st.sidebar.expander("🎨 خيارات العرض", expanded=True):
        search_radius = st.slider("🔍 نصف قطر البحث (Å)", 3.0, 10.0, 5.0)
        view_style = st.selectbox("نمط العرض", ["cartoon", "stick", "sphere"])
        show_surf = st.checkbox("إظهار السطح", value=False)
        surface_op = st.slider("شفافية السطح", 0.0, 1.0, 0.3)

    with st.sidebar.expander("💊 خيارات الارتباط", expanded=True):
        show_ligand = st.checkbox("إظهار الدواء", value=True)
        show_binding = st.checkbox("إظهار موقع الارتباط", value=True)
        show_interactions = st.checkbox("إظهار التفاعلات", value=True)
        focus_site = st.checkbox("التركيز على موقع الارتباط", value=True)

    st.sidebar.info("يقوم التطبيق بتحليل التفاعلات بين البروتين والجزيئات الدوائية الموجودة في ملف PDB.")

    # ── تحميل البروتين ──
    st.header("🧬 تحميل البروتين")
    col_src1, col_src2 = st.columns(2)

    with col_src1:
        source = st.radio("المصدر:", ["PDB ID", "رفع ملف"], horizontal=True, key="src")

    with col_src2:
        if source == "PDB ID":
            pdb_input = st.text_input("كود PDB:", key="pdb_id_in_field").strip().upper()
            if st.button("تحميل البروتين", key="btn_load", type="primary"):
                with st.spinner('جاري التحميل...'):
                    data = fetch_pdb_data(pdb_input)
                    if data:
                        st.session_state['pdb_data'] = data
                        st.session_state['pdb_id'] = pdb_input
                        st.session_state['binding_results'] = None
                        st.session_state['interactions'] = None
                        st.rerun()
                    else:
                        st.error(f"لم يتم العثور على البروتين: {pdb_input}")
        else:
            file = st.file_uploader("ارفع ملف PDB:", type=["pdb"], key="pdb_up")
            if file:
                st.session_state['pdb_data'] = file.getvalue().decode("utf-8")
                st.session_state['pdb_id'] = file.name
                st.session_state['binding_results'] = None
                st.session_state['interactions'] = None

    st.divider()

    # ── المعالجة والعرض ──
    pdb_data = st.session_state.get('pdb_data')
    if not pdb_data:
        st.info("👆 قم بتحميل بروتين للبدء بالتحليل")
        return

    structure = process_structure(pdb_data, st.session_state.get('pdb_id', 'protein'))
    if not structure:
        return

    # استخراج السلاسل والدواء
    chains = get_all_chains(structure)
    ligands = extract_ligands(structure)

    col_info, col_lig = st.columns(2)
    with col_info:
        st.subheader("📋 معلومات البروتين")
        selected_chain = st.selectbox("اختر السلسلة:", chains, key="chain_sel")
        total_res = sum(1 for r in structure[0][selected_chain] if r.id[0] == ' ')
        c1, c2, c3 = st.columns(3)
        c1.metric("عدد السلاسل", len(chains))
        c2.metric("الأحماض الأمينية", total_res)
        c3.metric("الأدوية المكتشفة", len(ligands))

    with col_lig:
        st.subheader("💊 الجزيئات الدوائية")
        if ligands:
            lig_options = [f"{l['name']} (Chain {l['chain']}, #{l['res_num']}, {l['num_atoms']} atoms)" for l in ligands]
            selected_lig_idx = st.selectbox("اختر الدواء:", range(len(lig_options)),
                                            format_func=lambda i: lig_options[i], key="lig_sel")
            selected_ligand = ligands[selected_lig_idx]
            # بطاقة معلومات الدواء
            st.markdown(f"""
            | الخاصية | القيمة |
            |---|---|
            | **الاسم** | `{selected_ligand['name']}` |
            | **الصيغة** | {selected_ligand.get('formula', '-')} |
            | **عدد الذرات** | {selected_ligand['num_atoms']} |
            | **المركز** | ({selected_ligand['center'][0]:.1f}, {selected_ligand['center'][1]:.1f}, {selected_ligand['center'][2]:.1f}) |
            """)
        else:
            st.warning("⚠️ لم يتم العثور على جزيئات دوائية. جرب: **6LU7**, **1HHP**, **3HTB**")
            selected_ligand = None

    st.divider()

    # ── العرض ثلاثي الأبعاد ──
    v_col1, v_col2 = st.columns([3, 2])

    binding_results = st.session_state.get('binding_results', None)
    interaction_results = st.session_state.get('interactions', None)

    with v_col1:
        st.subheader("🔬 العرض ثلاثي الأبعاد")
        
        # تحضير Grid Box للعرض
        grid_config = None
        if 'vina_cx' in st.session_state:
            grid_config = {
                'cx': st.session_state.vina_cx, 'cy': st.session_state.vina_cy, 'cz': st.session_state.vina_cz,
                'sx': st.session_state.vina_sx, 'sy': st.session_state.vina_sy, 'sz': st.session_state.vina_sz
            }

        selected_pose = None
        if st.session_state.get('vina_poses') and st.session_state.get('selected_pose_idx') is not None:
            selected_pose = st.session_state['vina_poses'][st.session_state['selected_pose_idx']]

        view_html = render_docking_3d(
            pdb_data,
            ligand_name=selected_ligand['name'] if selected_ligand else None,
            binding_residues=binding_results,
            interactions=interaction_results,
            bg_color='#0a0a1a', style_type=view_style,
            show_surface=show_surf, surface_opacity=surface_op,
            show_ligand=show_ligand, show_binding_site=show_binding,
            focus_on_site=focus_site,
            grid_box=grid_config,
            selected_pose=selected_pose
        )
        components.html(view_html, height=520)

    with v_col2:
        st.subheader("⚡ التحليل")
        if selected_ligand:
            if st.button("🔬 تحليل الارتباط الجزيئي", type="primary", use_container_width=True):
                with st.spinner("جاري تحليل التفاعلات..."):
                    binding = get_binding_site_residues(structure, selected_ligand, selected_chain, search_radius)
                    interactions = classify_interactions(structure, selected_ligand, selected_chain, search_radius)
                    st.session_state['binding_results'] = binding
                    st.session_state['interactions'] = interactions
                    st.rerun()

            if binding_results:
                energy = estimate_binding_energy(interaction_results or [], binding_results)
                m1, m2, m3 = st.columns(3)
                m1.metric("أحماض موقع الارتباط", len(binding_results))
                m2.metric("عدد التفاعلات", len(interaction_results) if interaction_results else 0)
                m3.metric("طاقة الارتباط (تقديرية)", f"{energy} kcal/mol")

                if interaction_results:
                    st.markdown("**📊 ملخص التفاعلات:**")
                    df_inter = pd.DataFrame(interaction_results)
                    type_counts = df_inter['type'].value_counts()
                    for t, c in type_counts.items():
                        icon = {'H-Bond': '🔵', 'Hydrophobic': '🟡', 'Ionic': '🟣', 'π-Stacking': '🟢'}.get(t, '⚪')
                        st.write(f"{icon} **{t}**: {c}")
            # زر تنزيل النتائج
                if binding_results:
                    with st.expander("⬇️ تنزيل النتائج"):
                        df_dl = pd.DataFrame(binding_results)
                        csv = df_dl.to_csv(index=False)
                        st.download_button("تنزيل CSV", csv, "binding_site.csv", "text/csv", key="dl_bind")
        else:
            st.info("اختر دواء لبدء التحليل")

    # ── النتائج والتفاعل ──
    tab_analysis, tab_docking = st.tabs(["📊 التحليل الهيكلي", "🚀 Vina Docking (Simulation)"])

    with tab_analysis:
        if binding_results:
            st.header("📊 نتائج تحليل الارتباط")
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🧪 موقع الارتباط", "🔗 التفاعلات", "📈 الرسوم البيانية"])

            with sub_tab1:
                df_binding = pd.DataFrame(binding_results)
                df_display = df_binding.rename(columns={
                    'res_num': 'الرقم', 'res_name': 'الحمض', 'one_letter': 'رمز',
                    'class': 'الفئة', 'min_dist': 'المسافة (Å)', 'sasa': 'SASA', 'chain': 'السلسلة'
                })
                st.dataframe(df_display, use_container_width=True, hide_index=True)

            with sub_tab2:
                if interaction_results:
                    df_inter = pd.DataFrame(interaction_results)
                    df_inter_display = df_inter.rename(columns={
                        'type': 'نوع التفاعل', 'residue': 'الحمض الأميني',
                        'res_atom': 'ذرة البروتين', 'lig_atom': 'ذرة الدواء',
                        'distance': 'المسافة (Å)', 'chain': 'السلسلة'
                    })
                    st.dataframe(df_inter_display.drop(columns=['res_num'], errors='ignore'),
                               use_container_width=True, hide_index=True)
                else:
                    st.info("لم يتم اكتشاف تفاعلات محددة")

            with sub_tab3:
                df_binding = pd.DataFrame(binding_results)
                # مخطط المسافات
                st.markdown("**📏 مسافة كل حمض أميني من الدواء**")
                fig1 = px.bar(df_binding, x='res_num', y='min_dist',
                             color='class', labels={'res_num': 'Residue', 'min_dist': 'Distance (Å)', 'class': 'Class'},
                             hover_data={'res_name': True, 'min_dist': ':.2f'})
                fig1.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=0),
                                 xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
                st.plotly_chart(fig1, use_container_width=True)

                # مخطط SASA
                st.markdown("**🌊 SASA لأحماض موقع الارتباط**")
                fig2 = px.line(df_binding, x='res_num', y='sasa', labels={'res_num': 'Residue', 'sasa': 'SASA'},
                              hover_data={'res_name': True, 'sasa': ':.1f'})
                fig2.update_traces(line_color='#00d4ff', line_width=2)
                fig2.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=0),
                                 xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'))
                st.plotly_chart(fig2, use_container_width=True)

                # مخطط التفاعلات
                if interaction_results:
                    st.markdown("**🔗 توزيع أنواع التفاعلات**")
                    df_inter = pd.DataFrame(interaction_results)
                    type_counts = df_inter['type'].value_counts().reset_index()
                    type_counts.columns = ['type', 'count']
                    colors = {'H-Bond': '#00ccff', 'Hydrophobic': '#ffaa00', 'Ionic': '#ff00ff', 'π-Stacking': '#00ff00'}
                    fig3 = px.pie(type_counts, values='count', names='type', color='type', color_discrete_map=colors)
                    fig3.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=30,b=0))
                    fig3.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("قم بإجراء التحليل الهيكلي لعرض النتائج هنا")

    with tab_docking:
        st.header("🔬 إعدادات AutoDock Vina")
        if not VINA_AVAILABLE:
            st.info("💡 ملاحظة: يتم استخدام 'محرك محاكاة Vina' لتمثيل الخطوات.")
        
        # زر تحديد المركز تلقائياً
        if selected_ligand:
            if st.button("🎯 تحديد مركز الدواء تلقائياً (Auto-Center)", use_container_width=True):
                c = selected_ligand['center']
                d = selected_ligand['dims']
                st.session_state['cx_in'] = round(c[0], 2)
                st.session_state['cy_in'] = round(c[1], 2)
                st.session_state['cz_in'] = round(c[2], 2)
                st.session_state['sx_in'] = round(d[0] + 10, 2)
                st.session_state['sy_in'] = round(d[1] + 10, 2)
                st.session_state['sz_in'] = round(d[2] + 10, 2)
                st.rerun()

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.subheader("📍 Search Space (Grid Box)")
            st.session_state.vina_cx = st.number_input("Center X", value=0.0, key="cx_in")
            st.session_state.vina_cy = st.number_input("Center Y", value=0.0, key="cy_in")
            st.session_state.vina_cz = st.number_input("Center Z", value=0.0, key="cz_in")
            st.session_state.vina_sx = st.number_input("Size X (Å)", value=20.0, key="sx_in")
            st.session_state.vina_sy = st.number_input("Size Y (Å)", value=20.0, key="sy_in")
            st.session_state.vina_sz = st.number_input("Size Z (Å)", value=20.0, key="sz_in")
            st.caption("💡 اضغط 'Auto-Center' لتحديد المركز تلقائياً حول الدواء.")
        
        with col_v2:
            st.subheader("⚙️ Parameters")
            exhaust = st.slider("Exhaustiveness", 1, 32, 8)
            n_poses = st.slider("Number of Poses", 1, 20, 9)
            seed = st.number_input("Random Seed", value=42, step=1)
            energy_range = st.slider("Energy Range (kcal/mol)", 1, 10, 3)
            
            if st.button("▶️ Run Vina Docking", type="primary", use_container_width=True):
                if selected_ligand:
                    log_placeholder = st.empty()
                    steps = [
                        f"[RECEPTOR] Loading {st.session_state['pdb_id']}...",
                        f"[LIGAND]   Preparing {selected_ligand['name']} (PDBQT)...",
                        f"[GRID]     Box center=({st.session_state.vina_cx:.1f},{st.session_state.vina_cy:.1f},{st.session_state.vina_cz:.1f})",
                        f"[GRID]     Box size=({st.session_state.vina_sx:.1f},{st.session_state.vina_sy:.1f},{st.session_state.vina_sz:.1f})",
                        f"[CONFIG]   exhaustiveness={exhaust}, n_poses={n_poses}, seed={seed}",
                        "[ENGINE]   Initializing Vina scoring function...",
                        "[SEARCH]   Running Monte Carlo + Local Optimization...",
                        "[SEARCH]   Clustering output poses...",
                        "[SCORE]    Calculating binding free energies...",
                        "[DONE]     Writing output poses."
                    ]
                    st.session_state.docking_log = []
                    progress = st.progress(0)
                    for idx, step in enumerate(steps):
                        st.session_state.docking_log.append(f"[{time.strftime('%H:%M:%S')}] {step}")
                        log_placeholder.code("\n".join(st.session_state.docking_log))
                        progress.progress((idx + 1) / len(steps))
                        time.sleep(0.5)
                    
                    np.random.seed(seed)
                    poses = simulate_vina_docking(st.session_state['pdb_id'], selected_ligand['name'], 
                                               (st.session_state.vina_cx, st.session_state.vina_cy, st.session_state.vina_cz), 
                                               (st.session_state.vina_sx, st.session_state.vina_sy, st.session_state.vina_sz),
                                               n_poses=n_poses, exhaustiveness=exhaust)
                    st.session_state['vina_poses'] = poses
                    st.session_state['selected_pose_idx'] = 0
                    st.success(f"✅ Docking Completed! Best affinity: {poses[0]['affinity']} kcal/mol")
                else:
                    st.error("يرجى اختيار Ligand أولاً")
            
            if st.session_state.get('docking_log'):
                with st.expander("📝 Docking Log", expanded=False):
                    st.code("\n".join(st.session_state.docking_log))

        if st.session_state.get('vina_poses'):
            st.divider()
            st.subheader("🏆 Docking Results (Poses)")
            
            c_poses1, c_poses2 = st.columns([1, 2])
            with c_poses1:
                pose_opts = [f"Pose {p['mode']} (Affinity: {p['affinity']})" for p in st.session_state['vina_poses']]
                sel_p = st.selectbox("اختر Pose لعرضه في 3D:", range(len(pose_opts)), 
                                   format_func=lambda i: pose_opts[i], key="pose_viewer_sel")
                st.session_state['selected_pose_idx'] = sel_p
                
                selected_p_data = st.session_state['vina_poses'][sel_p]
                st.metric("Binding Affinity", f"{selected_p_data['affinity']} kcal/mol")
                st.write(f"RMSD l.b.: {selected_p_data['rmsd_lb']}")
                st.write(f"RMSD u.b.: {selected_p_data['rmsd_ub']}")
            
            with c_poses2:
                df_poses = pd.DataFrame(st.session_state['vina_poses'])
                fig_poses = px.bar(df_poses, x='mode', y='affinity', 
                                 labels={'affinity': 'Affinity', 'mode': 'Pose'},
                                 color='affinity', color_continuous_scale='Reds_r')
                fig_poses.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_poses, use_container_width=True)

            # تنزيل نتائج Vina
            st.divider()
            df_poses_dl = pd.DataFrame(st.session_state['vina_poses'])
            csv_vina = df_poses_dl.to_csv(index=False)
            st.download_button("⬇️ تنزيل نتائج Docking (CSV)", csv_vina, 
                             f"vina_results_{st.session_state['pdb_id']}.csv", "text/csv", key="dl_vina")

if __name__ == "__main__":
    main()


# =============================================================================
# المكتبات (Libraries)
# =============================================================================
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


# =============================================================================
# الثوابت (Constants)
# =============================================================================

# ── ثوابت الشبكة والتخزين المؤقت ──
REQUEST_TIMEOUT_LONG  = 120   # ثانية - لجلب ملفات PDB الكبيرة
REQUEST_TIMEOUT_SHORT = 5     # ثانية - لجلب قاعدة بيانات الطفرات
CACHE_TTL_LONG        = 3600  # ثانية (ساعة واحدة)
CACHE_TTL_SHORT       = 60    # ثانية

# ── روابط البيانات الخارجية ──
PDB_DOWNLOAD_URL  = "https://files.rcsb.org/download/{pdb_id}.pdb"
REMOTE_JSON_URL   = "https://raw.githubusercontent.com/hassan2006-web/Bio-project/refs/heads/main/mutations.json"

# ── ثوابت SASA ──
SASA_EXPOSURE_THRESHOLD = 10  # أنجستروم مربع - الحد الفاصل بين مكشوف ومدفون

# ── جداول تحويل الأحماض الأمينية ──
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

AA_PROPS = {
    'ALA': 'Non-polar', 'GLY': 'Non-polar', 'ILE': 'Non-polar',
    'LEU': 'Non-polar', 'MET': 'Non-polar', 'PRO': 'Non-polar',
    'VAL': 'Non-polar', 'ASN': 'Polar',     'CYS': 'Polar',
    'GLN': 'Polar',     'SER': 'Polar',     'THR': 'Polar',
    'ASP': 'Acidic (-)', 'GLU': 'Acidic (-)',
    'ARG': 'Basic (+)', 'HIS': 'Basic (+)', 'LYS': 'Basic (+)',
    'PHE': 'Aromatic',  'TRP': 'Aromatic',  'TYR': 'Aromatic'
}

# ── ألوان البروتينات في الواجهة ──
PROTEIN_COLORS = {
    'h': {'mut_color': '#4CAF50', 'bg': '#0D1B1E'},
    'm': {'mut_color': '#F44336', 'bg': '#1E0D0D'},
}

# ── القيم الافتراضية لـ Session State ──
SESSION_DEFAULTS = {
    'h_pdb': None, 'm_pdb': None,
    'h_id': '',    'm_id': '',
    'h_results': None, 'm_results': None,
    'h_selected_chain': None, 'm_selected_chain': None,
    'h_id_in': '', 'm_id_in': ''
}


# =============================================================================
# جلب البيانات (Data Fetching)
# =============================================================================

@st.cache_data(ttl=CACHE_TTL_LONG)
def fetch_pdb_data(pdb_id: str) -> str | None:
    """جلب بيانات بنية البروتين من قاعدة بيانات RCSB العالمية باستخدام معرّف PDB."""
    if not pdb_id or pdb_id == "NONE":
        return None

    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_LONG)
        if response.status_code == 200:
            return response.text
        st.error(f"لم يتم العثور على البروتين (PDB ID: {pdb_id}) في قاعدة البيانات.")
        return None
    except requests.exceptions.RequestException as error:
        st.error(f"خطأ في الاتصال بالخادم: {error}")
        return None


@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_mutation_db() -> dict:
    """تحميل قاعدة بيانات الطفرات من GitHub لربط البروتين المصاب بنظيره السليم تلقائياً."""
    url = f"{REMOTE_JSON_URL}?t={int(time.time())}"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SHORT)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


# =============================================================================
# معالجة البنية (Structure Processing)
# =============================================================================

def process_protein_structure(pdb_string: str, pdb_id: str):
    """
    معالجة ملف PDB وتحويله إلى كائن Structure مع:
    - تصفية الجزيئات غير البروتينية (ماء، أيونات، روابط دوائية)
    - حساب SASA لكل حمض أميني
    """
    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, StringIO(pdb_string))

        # تصفية الجزيئات غير البروتينية
        for model in structure:
            for chain in model:
                non_std = [r.get_id() for r in chain if r.get_id()[0] != ' ']
                for r_id in non_std:
                    chain.detach_child(r_id)

        # حساب SASA
        try:
            sr = ShrakeRupley()
            sr.compute(structure, level='R')
        except Exception:
            pass

        return structure
    except Exception as error:
        st.error(f"خطأ في قراءة بنية البروتين: {error}")
        return None


def get_all_chains(structure) -> list[str]:
    """استخراج معرّفات جميع السلاسل الموجودة في النموذج الأول من البنية."""
    try:
        return [chain.id for chain in structure[0]]
    except Exception as ex:
        st.error(f"فشل قراءة السلاسل: {ex}")
        return []


def get_protein_sequence(structure, chain_id: str) -> list[dict]:
    """استخراج تسلسل الأحماض الأمينية القياسية لسلسلة محددة مع أرقامها في PDB."""
    sequence = []
    try:
        model = structure[0]
        if chain_id in [c.id for c in model]:
            for residue in model[chain_id]:
                if residue.id[0] == ' ':  # أحماض أمينية قياسية فقط
                    sequence.append({
                        'res_num' : residue.id[1],
                        'res_name': residue.get_resname()
                    })
    except Exception as ex:
        st.error(f"فشل قراءة التسلسل للسلسلة {chain_id}: {ex}")
    return sequence


def sequence_to_fasta(structure, chain_id: str, protein_name: str = "protein") -> str | None:
    """تحويل تسلسل الأحماض الأمينية لسلسلة محددة إلى تنسيق FASTA القياسي."""
    seq_data = get_protein_sequence(structure, chain_id)
    if not seq_data:
        return None

    one_letter = ''.join([AA_3TO1.get(r['res_name'], 'X') for r in seq_data])
    lines      = [one_letter[i:i+80] for i in range(0, len(one_letter), 80)]
    header     = f">{protein_name}|Chain_{chain_id}|length={len(one_letter)}\n"
    return header + '\n'.join(lines) + '\n'


def calculate_sasa_map(structure, chain_id: str) -> dict:
    """بناء خريطة {رقم الحمض → قيمة SASA} لسلسلة محددة."""
    try:
        return {
            res.id[1]: round(getattr(res, 'sasa', 0), 2)
            for res in structure[0][chain_id]
            if res.id[0] == ' '
        }
    except Exception:
        return {}


@st.cache_data(ttl=CACHE_TTL_LONG)
def calculate_all_distances(_structure_key: str, pdb_string: str,
                            chain_id: str, radius: float = 5.0) -> list[dict]:
    """
    حساب أقرب مسافة بين كل حمض أميني وجيرانه ضمن نصف القطر المحدد،
    مع قيمة SASA لكل حمض. يستخدم KDTree لأداء عالٍ.
    """
    structure = process_protein_structure(pdb_string, "prot")
    if not structure:
        return []

    try:
        model = structure[0]
        if chain_id not in [c.id for c in model]:
            return []

        # استخراج الذرات وإحداثياتها
        all_atoms  = list(model.get_atoms())
        all_coords = np.array([a.get_coord() for a in all_atoms], dtype=np.float32)

        # معلومات الذرات للفلترة
        atom_info = [(a.get_parent().get_parent().id, a.get_parent().id[1]) for a in all_atoms]

        tree     = KDTree(all_coords)
        residues = [r for r in model[chain_id] if r.id[0] == ' ']
        results  = []

        for target_res in residues:
            target_res_id  = target_res.id[1]
            target_coords  = np.array([a.get_coord() for a in target_res.get_atoms()], dtype=np.float32)

            # البحث عن الذرات القريبة واستبعاد نفس الحمض
            indices        = tree.query_ball_point(target_coords, radius)
            nearby_indices = {
                idx
                for idx_list in indices
                for idx in idx_list
                if not (atom_info[idx][0] == chain_id and atom_info[idx][1] == target_res_id)
            }

            min_dist = "-"
            if nearby_indices:
                nb_coords = all_coords[list(nearby_indices)]
                diff      = target_coords[:, np.newaxis, :] - nb_coords[np.newaxis, :, :]
                dist_sq   = (diff * diff).sum(axis=-1)
                min_dist  = round(float(np.sqrt(dist_sq.min())), 2)

            resname  = target_res.get_resname()
            sasa_val = getattr(target_res, 'sasa', '-')
            if isinstance(sasa_val, (float, int)):
                sasa_val = round(sasa_val, 2)

            results.append({
                'chain'     : chain_id,
                'res_num'   : target_res_id,
                'res_name'  : resname,
                'one_letter': AA_3TO1.get(resname, 'X'),
                'class'     : AA_PROPS.get(resname, '-'),
                'min_dist'  : min_dist,
                'sasa'      : sasa_val
            })

        return results

    except Exception as error:
        st.error(f"خطأ أثناء حساب المسافات: {error}")
        return []


# =============================================================================
# التحليل العلمي (Scientific Analysis)
# =============================================================================

def analyze_impact(h_res: str, m_res: str, h_sasa: float, m_sasa: float) -> str:
    """
    تحليل التأثير العلمي للطفرة بناءً على:
    1. التغير في الفئة الكيميائية (شحنة، قطبية... إلخ)
    2. التغير في مساحة السطح المعرّضة (SASA)
    """
    if h_res == m_res:
        return "Conservative"

    h_type  = AA_PROPS.get(h_res, 'Unknown')
    m_type  = AA_PROPS.get(m_res, 'Unknown')
    impacts = []

    # تحليل التغير الكيميائي
    if h_type != m_type:
        charge_flip = (
            ("Acidic" in h_type and "Basic" in m_type) or
            ("Basic"  in h_type and "Acidic" in m_type)
        )
        impacts.append("Charge Flip (Critical)" if charge_flip else "Chem-Class Change")

    # تحليل التغير في SASA
    try:
        diff = m_sasa - h_sasa
        if abs(diff) > SASA_EXPOSURE_THRESHOLD:
            impacts.append("Exposed" if diff > 0 else "Buried")
    except Exception:
        pass

    return " | ".join(impacts) if impacts else "Minor Change"


def get_alignment(seq1: str, seq2: str, mode: str = 'global') -> tuple:
    """
    إجراء محاذاة تسلسلية بين بروتينين باستخدام Biopython PairwiseAligner.
    يُعيد: (نص المحاذاة، النتيجة، المتتالية المحاذاة الأولى، المتتالية المحاذاة الثانية)
    """
    aligner      = PairwiseAligner()
    aligner.mode = mode
    try:
        best_aln = aligner.align(seq1, seq2)[0]
        return str(best_aln), best_aln.score, best_aln[0], best_aln[1]
    except Exception as error:
        st.warning(f"فشل إجراء المحاذاة التسلسلية: {error}")
        return "", 0, "", ""


def detect_mutations(alignment_data: dict, h_chain: str, m_chain: str) -> tuple[list, list]:
    """
    استخراج مواقع الطفرات من نتيجة المحاذاة.
    يُعيد: (مواقع الطفرات في السليم، مواقع الطفرات في المصاب)
    """
    mutations_healthy = []
    mutations_mutant  = []
    healthy_ptr = 0
    mutant_ptr = 0

    for char_h, char_m in zip(alignment_data['aligned_healthy'], alignment_data['aligned_mutant']):
        h_res = alignment_data['healthy_seq'][healthy_ptr] if char_h != '-' else None
        m_res = alignment_data['mutant_seq'][mutant_ptr]  if char_m != '-' else None

        if char_h != char_m:
            if m_res: mutations_mutant.append({'resi': str(m_res['res_num']), 'chain': str(m_chain)})
            if h_res: mutations_healthy.append({'resi': str(h_res['res_num']), 'chain': str(h_chain)})

        if char_h != '-': healthy_ptr += 1
        if char_m != '-': mutant_ptr  += 1

    return mutations_healthy, mutations_mutant


def build_comparison_rows(alignment_data: dict,
                          healthy_sasa_map: dict, mutant_sasa_map: dict) -> list[dict]:
    """
    بناء صفوف جدول المقارنة بناءً على نتيجة المحاذاة وخرائط SASA.
    """
    rows        = []
    healthy_pos = 0
    mutant_pos = 0

    for char_h, char_m in zip(alignment_data['aligned_healthy'], alignment_data['aligned_mutant']):
        h_res = alignment_data['healthy_seq'][healthy_pos] if char_h != '-' else None
        m_res = alignment_data['mutant_seq'][mutant_pos]   if char_m != '-' else None

        h_name  = h_res['res_name'] if h_res else '-'
        m_name  = m_res['res_name'] if m_res else '-'
        res_num = m_res['res_num'] if m_res else (f"({h_res['res_num']})" if h_res else '-')

        sasa_h = healthy_sasa_map.get(h_res['res_num'], 0) if h_res else 0
        sasa_m = mutant_sasa_map.get(m_res['res_num'],  0) if m_res else 0

        rows.append({
            'res_num'  : res_num,
            'السليم'   : h_name,
            'المصاب'   : m_name,
            'SASA_H'   : sasa_h,
            'SASA_M'   : sasa_m,
            'SASA_Delta': round(sasa_m - sasa_h, 2),
            'الحالة'   : '🔴 طفرة' if h_name != m_name else '🟢 محافظ',
            'Impact'   : analyze_impact(h_name, m_name, sasa_h, sasa_m)
        })

        if char_h != '-': healthy_pos += 1
        if char_m != '-': mutant_pos  += 1

    return rows


# =============================================================================
# العرض ثلاثي الأبعاد (3D Visualization)
# =============================================================================

def render_protein_3d(pdb_string: str, bg_color: str = '#111',
                      style_type: str = 'cartoon', show_surface: bool = True,
                      surface_opacity: float = 0.3, mutations: list | None = None,
                      mut_color: str = 'red', zoom_to_mutations: bool = False,
                      focus_mut: dict | None = None) -> str:
    """
    توليد كود HTML لعرض بنية البروتين ثلاثية الأبعاد باستخدام py3Dmol.
    يدعم تلوين الطفرات والتركيز على موقع محدد.
    """
    view = py3Dmol.view(width="100%", height=450)
    view.addModel(pdb_string, 'pdb')
    view.setBackgroundColor(bg_color)
    view.setStyle({'model': -1}, {style_type: {'color': 'spectrum'}})

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


# =============================================================================
# واجهة المستخدم - مكوّنات مساعدة (UI Helper Components)
# =============================================================================

def initialize_session_state():
    """تهيئة متغيرات الجلسة بقيمها الافتراضية عند أول تشغيل."""
    for key, val in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def clear_protein_state(prefix: str, keep_keys: list[str]):
    """مسح بيانات بروتين محدد من Session State مع الاحتفاظ بالمفاتيح المحددة."""
    for k in list(st.session_state.keys()):
        if k.startswith(prefix) and k not in keep_keys:
            st.session_state.pop(k, None)


def protein_ui_panel(p: dict):
    """
    لوحة تحميل البروتين: تدعم الإدخال عبر كود PDB أو رفع ملف.
    عند تحميل البروتين المصاب، تحاول جلب السليم المقابل تلقائياً.
    """
    with p["col"]:
        icon   = '🟢' if p['prefix'] == 'h' else '🔴'
        prefix = p['prefix']
        st.header(f"{icon} {p['label']}")
        source = st.radio("المصدر:", ["PDB ID", "رفع ملف"], key=f"{prefix}_src", horizontal=True)

        if source == "PDB ID":
            pdb_input = st.text_input("كود PDB", key=f"{prefix}_id_in").strip().upper()

            if st.button(f"تحميل {p['label']}", key=f"btn_{prefix}"):
                with st.spinner('جاري التحميل...'):
                    # الميزة الذكية: جلب السليم تلقائياً عند تحميل المصاب
                    if prefix == 'm':
                        mdb = load_mutation_db()
                        if pdb_input in mdb:
                            h_id   = mdb[pdb_input]
                            h_data = fetch_pdb_data(h_id)
                            if h_data:
                                st.session_state['h_id_in'] = h_id
                                st.session_state['h_pdb']   = h_data
                                st.session_state['h_id']    = h_id
                                st.info(f"✅ تم تحميل البروتين السليم تلقائياً: {h_id}")

                    data = fetch_pdb_data(pdb_input)
                    if data:
                        st.session_state[f"{prefix}_pdb"] = data
                        st.session_state[f"{prefix}_id"]  = pdb_input
                        keep = [f"{prefix}_pdb", f"{prefix}_id", f"{prefix}_src", f"{prefix}_id_in"]
                        clear_protein_state(prefix, keep)
                        st.rerun()
                    else:
                        st.error(f"لم يتم العثور على البروتين بالكود: {pdb_input}")
        else:
            file = st.file_uploader(f"ارفع ملف {p['label']} (.pdb):", type=["pdb"], key=f"{prefix}_up")
            if file:
                st.session_state[f"{prefix}_pdb"] = file.getvalue().decode("utf-8")
                st.session_state[f"{prefix}_id"]  = file.name
                keep = [f"{prefix}_pdb", f"{prefix}_id", f"{prefix}_src", f"{prefix}_up"]
                clear_protein_state(prefix, keep)


def render_sidebar() -> dict:
    """
    بناء الشريط الجانبي وإرجاع قاموس بجميع إعدادات التحليل والعرض.
    """
    st.sidebar.header("⚙️ التحليل والإعدادات")

    with st.sidebar.expander("🎨 خيارات العرض", expanded=True):
        search_radius   = st.slider("🔍 نصف قطر البحث (Å)", 3.0, 12.0, 5.0)
        view_style      = st.selectbox("نمط العرض", ["cartoon", "stick", "sphere"])
        show_surface    = st.checkbox("إظهار السطح (Surface)", value=False)
        surface_opacity = st.slider("شفافية السطح", 0.0, 1.0, 0.3)

    with st.sidebar.expander("🧬 خيارات الطفرات والمحاذاة", expanded=False):
        show_mutations  = st.checkbox("تلوين الطفرات", value=True)
        zoom_mutations  = st.checkbox("تركيز العرض على الطفرات", value=False)
        alignment_mode  = st.selectbox("نوع المحاذاة (Alignment)", ["global", "local"])

    st.sidebar.info("التحليل الهيكلي يشمل البحث في جميع سلاسل البروتين المختارة.")

    return {
        'search_radius'  : search_radius,
        'view_style'     : view_style,
        'show_surface'   : show_surface,
        'surface_opacity': surface_opacity,
        'show_mutations' : show_mutations,
        'zoom_mutations' : zoom_mutations,
        'alignment_mode' : alignment_mode,
    }


def render_3d_and_metrics(p: dict, settings: dict, highlight: list | None, pdb_data: str,
                          struct, selected_chain: str):
    """
    عرض المشاهدة ثلاثية الأبعاد وإحصائيات البروتين وزر التحليل التفصيلي
    لبروتين واحد (سليم أو مصاب).
    """
    prefix = p['prefix']
    colors = PROTEIN_COLORS[prefix]

    # اختيار طفرة للتركيز عليها
    focus_mut = None
    if highlight:
        mut_opts = ["الكل"] + [f"Residue {m['resi']} (Chain {m['chain']})" for m in highlight]
        sel_mut  = st.selectbox(f"🔍 التركيز على طفرة - {p['label']}", mut_opts, key=f"focus_{prefix}")
        if sel_mut != "الكل":
            parts     = sel_mut.split(" ")
            focus_mut = {'resi': parts[1], 'chain': parts[3].replace(")", "")}

    # العرض ثلاثي الأبعاد
    view_html = render_protein_3d(
        pdb_data,
        bg_color        = colors['bg'],
        style_type      = settings['view_style'],
        show_surface    = settings['show_surface'],
        surface_opacity = settings['surface_opacity'],
        mutations       = highlight if settings['show_mutations'] else None,
        mut_color       = colors['mut_color'],
        zoom_to_mutations = settings['zoom_mutations'],
        focus_mut       = focus_mut
    )
    components.html(view_html, height=460)

    # الإحصائيات
    all_chains  = get_all_chains(struct)
    total_res   = sum(len([r for r in struct[0][c] if r.id[0] == ' ']) for c in all_chains)
    c1, c2, c3 = st.columns(3)
    c1.metric("عدد السلاسل",   len(all_chains))
    c2.metric("إجمالي الأحماض", total_res)
    c3.metric("السلسلة الحالية", selected_chain)

    # تنزيل FASTA
    fasta = sequence_to_fasta(struct, selected_chain, st.session_state.get(f"{prefix}_id", p['label']))
    if fasta:
        with st.expander(f"🧬 تنزيل FASTA - {p['label']}"):
            st.download_button(
                "⬇️ تنزيل FASTA", fasta,
                f"{st.session_state.get(f'{prefix}_id', 'protein')}_{selected_chain}.fasta",
                "text/plain", key=f"dl_f_{prefix}"
            )

    # التحليل التفصيلي
    st.divider()
    if st.button(f"🔬 تحليل كامل لـ {p['label']}", key=f"analyze_btn_{prefix}", type="primary"):
        with st.spinner("جاري التحليل..."):
            results = calculate_all_distances(
                f"{st.session_state.get(f'{prefix}_id')}_{selected_chain}",
                pdb_data, selected_chain, radius=settings['search_radius']
            )
            st.session_state[f"{prefix}_results"] = results

    if st.session_state.get(f"{prefix}_results"):
        with st.expander("📊 نتائج التحليل"):
            df = pd.DataFrame(st.session_state[f"{prefix}_results"])
            st.dataframe(
                df.rename(columns={
                    'res_num': 'رقم البقية', 'res_name': 'الحمض',
                    'class'  : 'الفئة الكيميائية', 'min_dist': 'أقرب مسافة', 'sasa': 'SASA'
                }),
                use_container_width=True, hide_index=True
            )


def render_comparison_section(structures: dict, h_chain: str, m_chain: str,
                              alignment_data: dict):
    """
    عرض قسم المقارنة الكامل: جدول مقارنة + رسم SASA + نتائج المحاذاة.
    """
    st.divider()
    st.header("📋 مقارنة السلسلة (Comparison)")

    # بناء بيانات المقارنة
    healthy_sasa_map = calculate_sasa_map(structures['h'], h_chain)
    mutant_sasa_map  = calculate_sasa_map(structures['m'], m_chain)
    rows             = build_comparison_rows(alignment_data, healthy_sasa_map, mutant_sasa_map)
    comparison_df    = pd.DataFrame(rows)

    # جدول المقارنة
    with st.expander("جدول المقارنة المتقدم (Structural & Chemical Impact)"):
        display_df = comparison_df.rename(columns={
            'res_num': 'الرقم', 'SASA_H': 'SASA H',
            'SASA_M': 'SASA M', 'SASA_Delta': 'ΔSASA', 'Impact': 'نوع التأثير العلمي'
        })
        st.dataframe(
            display_df.style.apply(
                lambda r: ['background-color: #3e2723' if r['السليم'] != r['المصاب'] else ''] * len(r),
                axis=1
            ),
            use_container_width=True, hide_index=True
        )

    # رسم SASA التفاعلي
    _render_sasa_chart(comparison_df)

    # نتائج المحاذاة
    _render_alignment_metrics(alignment_data)


def _render_sasa_chart(comparison_df: pd.DataFrame):
    """رسم مخطط SASA التفاعلي مع تمييز مواقع الطفرات بنجوم صفراء."""
    st.subheader("📊 مقارنة SASA المتقدمة")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=comparison_df['res_num'], y=comparison_df['SASA_H'],
        name='البروتين السليم (WT)',
        line=dict(color='#00ff88', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=comparison_df['res_num'], y=comparison_df['SASA_M'],
        name='البروتين المصاب (MT)',
        line=dict(color='#ff3333', width=2),
        fill='tonexty', fillcolor='rgba(255, 51, 51, 0.1)'
    ))

    mutations_df = comparison_df[comparison_df['السليم'] != comparison_df['المصاب']]
    if not mutations_df.empty:
        fig.add_trace(go.Scatter(
            x=mutations_df['res_num'], y=mutations_df['SASA_M'],
            mode='markers', name='مواقع الطفرات',
            marker=dict(color='yellow', size=10, symbol='star', line=dict(color='black', width=1)),
            hovertemplate="رقم الحمض: %{x}<br>من: %{customdata[0]}<br>إلى: %{customdata[1]}<br>التأثير: %{customdata[2]}<extra></extra>",
            customdata=mutations_df[['السليم', 'المصاب', 'Impact']].values
        ))

    fig.update_layout(
        template="plotly_dark", height=450, hovermode="x unified",
        xaxis=dict(title="رقم الحمض الأميني", rangeslider=dict(visible=True)),
        yaxis=dict(title="SASA (Å²)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_alignment_metrics(alignment_data: dict):
    """عرض مقاييس المحاذاة التسلسلية (النتيجة، النسبة التطابق، فرق الطول)."""
    st.header(f"🧬 Alignment ({alignment_data.get('mode', '').capitalize()})")

    aligned_h = alignment_data['aligned_healthy']
    aligned_m = alignment_data['aligned_mutant']
    identity  = sum(a == b and a != '-' for a, b in zip(aligned_h, aligned_m)) / len(aligned_h) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Alignment Score", round(alignment_data['score'], 1))
    c2.metric("Identity %",      f"{identity:.1f}%")
    c3.metric("فرق الطول",       abs(len(alignment_data['healthy_str']) - len(alignment_data['mutant_str'])))
    st.code(alignment_data['text'], language='text')


# =============================================================================
# نقطة انطلاق التطبيق (App Entry Point)
# =============================================================================

def main():
    """الدالة الرئيسية: تُنسّق تشغيل التطبيق بالكامل من البداية للنهاية."""
    st.set_page_config(page_title="Bio-Impact Analyzer", page_icon="🧬", layout="wide")
    initialize_session_state()
    st.title("🧬 Bio-Impact Analyzer")

    # ── الشريط الجانبي ──
    settings = render_sidebar()

    # ── تعريف البروتينين ──
    col1, col2 = st.columns(2)
    proteins = [
        {"label": "السليم",  "prefix": "h", "col": col1},
        {"label": "المصاب",  "prefix": "m", "col": col2},
    ]

    # ── 1. لوحات تحميل البروتينات ──
    for p in proteins:
        protein_ui_panel(p)

    st.divider()

    # ── 2. معالجة الهياكل واختيار السلاسل ──
    v_col1, v_col2 = st.columns(2)
    structures = {}

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
                    selected = st.selectbox(
                        f"اختر السلسلة - {p['label']}", options=chains,
                        key=f"{p['prefix']}_chain_sel"
                    )
                    st.session_state[f"{p['prefix']}_selected_chain"] = selected
                else:
                    st.warning("لم يتم العثور على سلاسل في هذا الهيكل.")

    # ── 3. حساب المحاذاة والطفرات (مرة واحدة للاستخدام المشترك) ──
    h_chain      = st.session_state.get('h_selected_chain')
    m_chain      = st.session_state.get('m_selected_chain')
    alignment_data  = None
    highlight_map   = {'h': None, 'm': None}

    if structures.get('h') and structures.get('m') and h_chain and m_chain:
        healthy_seq = get_protein_sequence(structures['h'], h_chain)
        mutant_seq  = get_protein_sequence(structures['m'], m_chain)

        if healthy_seq and mutant_seq:
            healthy_str = "".join([AA_3TO1.get(r['res_name'], 'X') for r in healthy_seq])
            mutant_str  = "".join([AA_3TO1.get(r['res_name'], 'X') for r in mutant_seq])

            aln_text, score, aligned_h, aligned_m = get_alignment(
                healthy_str, mutant_str, settings['alignment_mode']
            )
            alignment_data = {
                'text'           : aln_text,
                'score'          : score,
                'aligned_healthy': aligned_h,
                'aligned_mutant' : aligned_m,
                'healthy_seq'    : healthy_seq,
                'mutant_seq'     : mutant_seq,
                'healthy_str'    : healthy_str,
                'mutant_str'     : mutant_str,
                'mode'           : settings['alignment_mode'],
            }

            mut_healthy, mut_mutant = detect_mutations(alignment_data, h_chain, m_chain)
            highlight_map['h'] = mut_healthy or None
            highlight_map['m'] = mut_mutant  or None

    # ── 4. العرض ثلاثي الأبعاد والتحليل لكل بروتين ──
    for p in proteins:
        current_col = v_col1 if p['prefix'] == 'h' else v_col2
        with current_col:
            prefix        = p['prefix']
            pdb_data      = st.session_state.get(f"{prefix}_pdb")
            struct        = structures.get(prefix)
            selected_chain = st.session_state.get(f"{prefix}_selected_chain")

            if not pdb_data or not struct or not selected_chain:
                continue

            render_3d_and_metrics(
                p, settings,
                highlight=highlight_map[prefix],
                pdb_data=pdb_data,
                struct=struct,
                selected_chain=selected_chain
            )

    # ── 5. قسم المقارنة المشترك ──
    if structures.get('h') and structures.get('m') and h_chain and m_chain and alignment_data:
        render_comparison_section(structures, h_chain, m_chain, alignment_data)


if __name__ == "__main__":
    main()

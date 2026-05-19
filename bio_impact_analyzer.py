# =============================================================================
#  Bio-Impact Analyzer (محلل التأثير البيولوجي للطفرات البروتينية)
#  تطبيق تفاعلي متقدم لتحليل بنية البروتينات ثلاثية الأبعاد وقياس تأثير الطفرات
# =============================================================================

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
# تستخدم للتحكم في أوقات الاستجابة وصلاحية التخزين المؤقت لتسريع التطبيق وتحسين استقراره
REQUEST_TIMEOUT_LONG  = 120   # ثانية - مهلة جلب ملفات البنية ثلاثية الأبعاد الكبيرة من الخادم
REQUEST_TIMEOUT_SHORT = 5     # ثانية - مهلة جلب ملف الطفرات من مستودع GitHub
CACHE_TTL_LONG        = 3600  # ثانية (ساعة واحدة) - مدة صلاحية ذاكرة الكاش لملفات البنية PDB
CACHE_TTL_SHORT       = 60    # ثانية - مدة صلاحية ذاكرة الكاش لقائمة الطفرات

# ── روابط البيانات الخارجية ──
# الرابط المباشر لتحميل ملفات البنية ثلاثية الأبعاد (PDB) من بنك بيانات البروتينات العالمي RCSB PDB
PDB_DOWNLOAD_URL  = "https://files.rcsb.org/download/{pdb_id}.pdb"
# رابط ملف الطفرات بصيغة JSON على مستودع GitHub للمطابقة التلقائية بين المصاب والسليم
REMOTE_JSON_URL   = "https://raw.githubusercontent.com/hassan2006-web/Bio-project/refs/heads/main/mutations.json"

# ── ثوابت SASA (مساحة السطح المعرضة للمذيب) ──
# الحد الأدنى للتغير في مساحة SASA (بالأنجستروم المربع) لاعتبار الحمض الأميني مكشوفاً أو مدفوناً بعد الطفرة
SASA_EXPOSURE_THRESHOLD = 10  # أنجستروم مربع (Å²)

# ── جداول تحويل الأحماض الأمينية ──
# جدول لتحويل أسماء الأحماض الأمينية من الاختصار ثلاثي الأحرف إلى الرمز القياسي ذي الحرف الواحد
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# تصنيف الأحماض الأمينية بناءً على خصائصها الكيميائية لتحديد أثر الطفرة على البنية (تغير الفئة الكيميائية)
AA_PROPS = {
    'ALA': 'Non-polar', 'GLY': 'Non-polar', 'ILE': 'Non-polar',
    'LEU': 'Non-polar', 'MET': 'Non-polar', 'PRO': 'Non-polar',
    'VAL': 'Non-polar', 'ASN': 'Polar',     'CYS': 'Polar',
    'GLN': 'Polar',     'SER': 'Polar',     'THR': 'Polar',
    'ASP': 'Acidic (-)', 'GLU': 'Acidic (-)',
    'ARG': 'Basic (+)', 'HIS': 'Basic (+)', 'LYS': 'Basic (+)',
    'PHE': 'Aromatic',  'TRP': 'Aromatic',  'TYR': 'Aromatic'
}

# ── ألوان وسمات الواجهة ثلاثية الأبعاد ──
# الألوان والخلفيات المميزة للبروتين السليم (WT) والبروتين المصاب (MT) لسهولة التمييز البصري
PROTEIN_COLORS = {
    'h': {'mut_color': '#4CAF50', 'bg': '#0D1B1E'},  # السليم (أخضر)
    'm': {'mut_color': '#F44336', 'bg': '#1E0D0D'},  # المصاب (أحمر)
}

# ── القيم الافتراضية لمتغيرات الجلسة (Session State) ──
# تهيئة جميع القيم الافتراضية لمنع الأخطاء البرمجية أثناء الرندر الأولي للتطبيق
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
    """
    شرح التابع:
    يقوم بالاتصال بخوادم بنك البروتينات العالمي (RCSB PDB) وجلب ملف البيانات الهيكلية للبروتين المطلوب.
    
    المدخلات:
    - pdb_id: معرّف البروتين الفريد (مكون من 4 رموز، مثل: 1A2B).
    
    المخرجات:
    - محتوى ملف الـ PDB النصي في حال النجاح، أو None في حال حدوث أي خطأ بالشبكة.
    """
    if not pdb_id or pdb_id == "NONE":
        return None

    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_LONG)
        if response.status_code == 200:
            return response.text
        st.error(f"لم يتم العثور على البروتين (PDB ID: {pdb_id}) في قاعدة البيانات العالمية.")
        return None
    except requests.exceptions.RequestException as error:
        st.error(f"خطأ في الاتصال بالخادم: {error}")
        return None


@st.cache_data(ttl=CACHE_TTL_SHORT)
def load_mutation_db() -> dict:
    """
    شرح التابع:
    يجلب خريطة الطفرات (ملف JSON) من مستودع GitHub لتمكين ميزة التحميل التلقائي 
    للبروتين السليم المقابل فور إدخال كود البروتين المصاب.
    
    المخرجات:
    - قاموس (Dictionary) يحتوي على الروابط بين معرفات البروتينات المصابة والسليمة.
    """
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
    شرح التابع:
    يقوم بتحويل النص الخام لملف PDB إلى كائن هيكلي ثلاثي الأبعاد (Structure Object) مع:
    1. فلترة وتصفية الجزيئات غير البروتينية (مثل جزيئات الماء والأيونات الحرة).
    2. حساب مساحة السطح المعرضة للمذيب (SASA) لكل حمض أميني في الهيكل باستخدام خوارزمية ShrakeRupley.
    
    المدخلات:
    - pdb_string: النص الكامل لملف PDB.
    - pdb_id: اسم أو معرف البروتين.
    
    المخرجات:
    - كائن Structure مهيأ ومحسوب له الـ SASA، أو None في حال الفشل.
    """
    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, StringIO(pdb_string))

        # فلترة الجزيئات غير البروتينية (الماء، الأيونات والروابط المضافة)
        # حيث يتم فحص المعرف فإذا لم يكن حمضاً أمينياً قياسياً يتم إزالته من السلسلة
        for model in structure:
            for chain in model:
                non_std = [r.get_id() for r in chain if r.get_id()[0] != ' ']
                for r_id in non_std:
                    chain.detach_child(r_id)

        # حساب مساحة السطح المعرضة للمذيب (SASA - Solvent Accessible Surface Area)
        # باستخدام خوارزمية Shrake-Rupley التي تمرر نموذجاً كروياً حول الذرات لمحاكاة جزيئات المذيب
        try:
            sr = ShrakeRupley()
            sr.compute(structure, level='R')  # الحساب على مستوى الحمض الأميني (Residue)
        except Exception:
            pass

        return structure
    except Exception as error:
        st.error(f"خطأ في قراءة بنية البروتين: {error}")
        return None


def get_all_chains(structure) -> list[str]:
    """
    شرح التابع:
    يستخرج معرّفات جميع السلاسل الببتيدية (Chains) المتواجدة في النموذج الأول للبنية البروتينية.
    
    المدخلات:
    - structure: كائن بنية البروتين.
    
    المخرجات:
    - قائمة برموز السلاسل المتوفرة (مثل: ['A', 'B']).
    """
    try:
        return [chain.id for chain in structure[0]]
    except Exception as ex:
        st.error(f"فشل قراءة السلاسل: {ex}")
        return []


def get_protein_sequence(structure, chain_id: str) -> list[dict]:
    """
    شرح التابع:
    يستخرج تسلسل الأحماض الأمينية القياسية لسلسلة محددة مع رقم كل حمض في الهيكل.
    
    المدخلات:
    - structure: كائن بنية البروتين.
    - chain_id: رمز السلسلة المطلوب استخراج تسلسلها.
    
    المخرجات:
    - قائمة قواميس تحتوي على رقم الحمض الأميني واسمه الثلاثي (مثال: [{'res_num': 1, 'res_name': 'ALA'}]).
    """
    sequence = []
    try:
        model = structure[0]
        if chain_id in [c.id for c in model]:
            for residue in model[chain_id]:
                if residue.id[0] == ' ':  # تصفية الأحماض الأمينية القياسية فقط وتجاهل جزيئات الماء والأيونات
                    sequence.append({
                        'res_num' : residue.id[1],
                        'res_name': residue.get_resname()
                    })
    except Exception as ex:
        st.error(f"فشل قراءة التسلسل للسلسلة {chain_id}: {ex}")
    return sequence


def sequence_to_fasta(structure, chain_id: str, protein_name: str = "protein") -> str | None:
    """
    شرح التابع:
    يحول تسلسل الأحماض الأمينية إلى صيغة FASTA القياسية (المستخدمة دولياً في تبادل وقراءة البيانات الحيوية).
    
    المدخلات:
    - structure: كائن بنية البروتين.
    - chain_id: رمز السلسلة المستهدفة.
    - protein_name: اسم البروتين لكتابته في رأس الملف.
    
    المخرجات:
    - نص بصيغة FASTA جاهز للتحميل أو العرض.
    """
    seq_data = get_protein_sequence(structure, chain_id)
    if not seq_data:
        return None

    # تحويل رموز 3 أحرف إلى حرف واحد
    one_letter = ''.join([AA_3TO1.get(r['res_name'], 'X') for r in seq_data])
    # تنسيق الأسطر لكي لا تتجاوز 80 حرفاً لكل سطر
    lines      = [one_letter[i:i+80] for i in range(0, len(one_letter), 80)]
    header     = f">{protein_name}|Chain_{chain_id}|length={len(one_letter)}\n"
    return header + '\n'.join(lines) + '\n'


def calculate_sasa_map(structure, chain_id: str) -> dict:
    """
    شرح التابع:
    يبني خريطة سريعة لربط رقم كل حمض أميني بقيمة مساحة سطحه المعرضة للمذيب (SASA).
    
    المدخلات:
    - structure: كائن بنية البروتين.
    - chain_id: رمز السلسلة الببتيدية.
    
    المخرجات:
    - قاموس يربط الرقم بالـ SASA (مثال: {1: 45.2, 2: 12.0}).
    """
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
    شرح التابع:
    يقوم بحساب أقرب مسافة بين كل حمض أميني وجيرانه في البنية ثلاثية الأبعاد 
    باستخدام خوارزمية شجرة البحث KDTree لتحقيق أداء وسرعة حسابية فائقة.
    
    المدخلات:
    - _structure_key: مفتاح فريد للتخزين المؤقت (الكاش).
    - pdb_string: نص ملف PDB.
    - chain_id: رمز السلسلة المستهدفة.
    - radius: نصف قطر البحث (بالأنجستروم) لتحديد الجيران القريبين.
    
    المخرجات:
    - قائمة تحتوي على الخصائص الهيكلية والبيئية لكل حمض أميني (المسافة، SASA، الفئة الكيميائية، إلخ).
    """
    structure = process_protein_structure(pdb_string, "prot")
    if not structure:
        return []

    try:
        model = structure[0]
        if chain_id not in [c.id for c in model]:
            return []

        # استخراج إحداثيات ذرات البروتين في الفضاء ثلاثي الأبعاد
        all_atoms  = list(model.get_atoms())
        all_coords = np.array([a.get_coord() for a in all_atoms], dtype=np.float32)

        # حفظ معلومات السلسلة ورقم الحمض لكل ذرة لتجنب مطابقة الحمض مع نفسه
        atom_info = [(a.get_parent().get_parent().id, a.get_parent().id[1]) for a in all_atoms]

        # بناء شجرة البحث الفراغي KDTree للبحث السريع عن أقرب الجيران
        tree     = KDTree(all_coords)
        residues = [r for r in model[chain_id] if r.id[0] == ' ']
        results  = []

        for target_res in residues:
            target_res_id  = target_res.id[1]
            target_coords  = np.array([a.get_coord() for a in target_res.get_atoms()], dtype=np.float32)

            # البحث عن كافة الذرات الواقعة ضمن نصف القطر المحدد واستبعاد ذرات نفس الحمض الأميني
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
                # حساب مصفوفة المسافات البينية بالاعتماد على الجبر الخطي لتسريع العملية
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
        st.error(f"خطأ أثناء حساب المسافات الفراغية: {error}")
        return []


# =============================================================================
# التحليل العلمي (Scientific Analysis)
# =============================================================================

def analyze_impact(h_res: str, m_res: str, h_sasa: float, m_sasa: float) -> str:
    """
    شرح التابع:
    يقوم بتحليل وتقييم الأثر البيولوجي والهيكلي للطفرة بناءً على معيارين رئيسيين:
    1. التغير في الفئة الكيميائية للأحماض الأمينية (مثال: من حمض شحنته سالبة إلى موجبة "Charge Flip").
    2. التغير في مساحة السطح المعرضة للمذيب (SASA) لتحديد ما إذا كان الحمض قد أصبح مكشوفاً (Exposed) أو مدفوناً (Buried) بالداخل.
    
    المدخلات:
    - h_res: اسم الحمض السليم (WT).
    - m_res: اسم الحمض المصاب (MT).
    - h_sasa: مساحة سطح الحمض السليم المعرضة للمذيب.
    - m_sasa: مساحة سطح الحمض المصاب المعرضة للمذيب.
    
    المخرجات:
    - نص وصفي يوضح طبيعة التأثير العلمي للطفرة (مثال: "Chem-Class Change | Buried").
    """
    if h_res == m_res:
        return "Conservative"  # طفرة محافظة (لم يتغير الحمض)

    h_type  = AA_PROPS.get(h_res, 'Unknown')
    m_type  = AA_PROPS.get(m_res, 'Unknown')
    impacts = []

    # 1. تحليل التغير الكيميائي
    if h_type != m_type:
        # التحقق مما إذا حدث انقلاب في الشحنة الكيميائية (خطير جداً بيولوجياً)
        charge_flip = (
            ("Acidic" in h_type and "Basic" in m_type) or
            ("Basic"  in h_type and "Acidic" in m_type)
        )
        impacts.append("Charge Flip (Critical)" if charge_flip else "Chem-Class Change")

    # 2. تحليل التغير في SASA
    try:
        diff = m_sasa - h_sasa
        if abs(diff) > SASA_EXPOSURE_THRESHOLD:
            # إذا زادت المساحة المعرضة يصبح مكشوفاً، وإذا نقصت يصبح مدفوناً
            impacts.append("Exposed" if diff > 0 else "Buried")
    except Exception:
        pass

    return " | ".join(impacts) if impacts else "Minor Change"


def get_alignment(seq1: str, seq2: str, mode: str = 'global') -> tuple:
    """
    شرح التابع:
    إجراء عملية محاذاة تسلسلية (Sequence Alignment) بين تسلسلي البروتين السليم والمصاب 
    لمعرفة الموضع الدقيق لكل طفرة أو فجوة (Gap) باستخدام Biopython PairwiseAligner.
    
    المدخلات:
    - seq1: تسلسل البروتين السليم بحروف مفردة.
    - seq2: تسلسل البروتين المصاب بحروف مفردة.
    - mode: نمط المحاذاة ('global' للمحاذاة الشاملة بكامل الطول، أو 'local' للمحاذاة الموضعية).
    
    المخرجات:
    - صف (Tuple) يحتوي على: (نص المحاذاة التوضيحي، نتيجة المطابقة الإجمالية Score، المتتالية الأولى المحاذية، المتتالية الثانية المحاذية).
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
    شرح التابع:
    يقارن تسلسلي البروتين المحاذيين لاستخراج وتحديد مواقع الطفرات الدقيقة (المواقع التي لا تتطابق فيها الأحماض).
    
    المدخلات:
    - alignment_data: قاموس يحتوي على بيانات المحاذاة والتسلسل المحاذي لكلا الطرفين.
    - h_chain: السلسلة المختارة للبروتين السليم.
    - m_chain: السلسلة المختارة للبروتين المصاب.
    
    المخرجات:
    - صف يحتوي على قائمتين: (مواقع الطفرات في البروتين السليم، مواقع الطفرات في البروتين المصاب).
    """
    mutations_healthy = []
    mutations_mutant  = []
    healthy_ptr = 0
    mutant_ptr = 0

    # المرور عبر تسلسل الأحماض المحاذاة حرفاً بحرف
    for char_h, char_m in zip(alignment_data['aligned_healthy'], alignment_data['aligned_mutant']):
        h_res = alignment_data['healthy_seq'][healthy_ptr] if char_h != '-' else None
        m_res = alignment_data['mutant_seq'][mutant_ptr]  if char_m != '-' else None

        # إذا اختلف الحرفان، فهناك طفرة أو فجوة استبدالية
        if char_h != char_m:
            if m_res: mutations_mutant.append({'resi': str(m_res['res_num']), 'chain': str(m_chain)})
            if h_res: mutations_healthy.append({'resi': str(h_res['res_num']), 'chain': str(h_chain)})

        # تحريك المؤشرات فقط للأحماض الفعلية وتخطي الفجوات (Gaps "-")
        if char_h != '-': healthy_ptr += 1
        if char_m != '-': mutant_ptr  += 1

    return mutations_healthy, mutations_mutant


def build_comparison_rows(alignment_data: dict,
                          healthy_sasa_map: dict, mutant_sasa_map: dict) -> list[dict]:
    """
    شرح التابع:
    يقوم ببناء بيانات جدول المقارنة المتقدم سطراً بسطر بدمج تسلسل المحاذاة مع قيم الـ SASA والتأثير العلمي.
    
    المدخلات:
    - alignment_data: نتائج عملية المحاذاة.
    - healthy_sasa_map: خريطة SASA للبروتين السليم.
    - mutant_sasa_map: خريطة SASA للبروتين المصاب.
    
    المخرجات:
    - قائمة من القواميس يمثل كل منها صفاً تفصيلياً جاهزاً للتحويل إلى DataFrame وعرضه للمستخدم.
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
    شرح التابع:
    توليد كود HTML تفاعلي متكامل لعرض البنية ثلاثية الأبعاد للبروتين باستخدام مكتبة py3Dmol الرائعة.
    يدعم تلوين الطفرات، التركيز على حمض محدد، وإظهار السطح الخارجي للبروتين.
    
    المدخلات:
    - pdb_string: نص ملف PDB للبروتين.
    - bg_color: لون خلفية مشهد العرض.
    - style_type: نمط العرض الهيكلي (cartoon، stick، أو sphere).
    - show_surface: تفعيل/إلغاء إظهار الغلاف الخارجي للبروتين.
    - surface_opacity: درجة شفافية السطح.
    - mutations: قائمة بمواضع الطفرات لتلوينها بشكل مميز.
    - mut_color: لون الطفرات المميز.
    - zoom_to_mutations: تركيز الكاميرا وتقريبها تلقائياً على كل الطفرات.
    - focus_mut: تركيز الكاميرا وتقريبها على حمض أميني واحد بعينه.
    
    المخرجات:
    - كود HTML جاهز للحقن في صفحة Streamlit لعرض المشهد التفاعلي.
    """
    view = py3Dmol.view(width="100%", height=450)
    view.addModel(pdb_string, 'pdb')
    view.setBackgroundColor(bg_color)
    view.setStyle({'model': -1}, {style_type: {'color': 'spectrum'}})

    # رسم الغلاف الخارجي للبروتين بالاعتماد على مساحة السطح المعرضة للمذيب
    if show_surface:
        view.addSurface(py3Dmol.SAS, {'opacity': surface_opacity, 'color': '#FFC107'})

    # تمييز وتلوين الطفرات بالهياكل العصوية (Sticks) والكروية (Spheres)
    if mutations:
        for mut in mutations:
            view.addStyle(mut, {style_type: {'color': mut_color}})
            view.addStyle(mut, {'stick'  : {'colorscheme': 'yellowCarbon', 'radius': 0.3}})
            view.addStyle(mut, {'sphere' : {'color': mut_color, 'radius': 1.2}})

    # التحكم بتركيز وتقريب الكاميرا (Zoom)
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
    """
    شرح التابع:
    يقوم بتهيئة المتغيرات الأساسية لجلسة عمل المستخدم (Session State) عند أول تشغيل للتطبيق.
    مهم جداً لمنع حدوث أخطاء عدم العثور على المتغيرات (KeyError).
    """
    for key, val in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def clear_protein_state(prefix: str, keep_keys: list[str]):
    """
    شرح التابع:
    يقوم بمسح البيانات الهيكلية والتحليلية القديمة لبروتين محدد من الذاكرة 
    عندما يقرر المستخدم تحميل بروتين جديد، مع الإبقاء على المتغيرات المستهدفة فقط.
    """
    for k in list(st.session_state.keys()):
        if k.startswith(prefix) and k not in keep_keys:
            st.session_state.pop(k, None)


def protein_ui_panel(p: dict):
    """
    شرح التابع:
    يبني لوحة التحكم التفاعلية الخاصة بتحميل وعرض البروتين (سواء كان السليم WT أو المصاب MT).
    تتيح الاختيار بين الإدخال المباشر لكود PDB أو رفع ملف من الكمبيوتر المحلي.
    
    *ملاحظة هامة:* تم تطبيق إصلاح برمجي هنا بفصل متغير الحالة `f"{prefix}_id_in"` عن مفتاح أداة الإدخال 
    لتجنب استثناء Streamlit الشهير وتسهيل التحميل التلقائي للبروتين السليم عند إدخال المصاب.
    """
    with p["col"]:
        icon   = '🟢' if p['prefix'] == 'h' else '🔴'
        prefix = p['prefix']
        st.header(f"{icon} {p['label']}")
        source = st.radio("المصدر:", ["PDB ID", "رفع ملف"], key=f"{prefix}_src", horizontal=True)

        if source == "PDB ID":
            # ── إصلاح استثناء Streamlit (StreamlitAPIException) ──
            # نستخدم متغيراً في session_state لحفظ القيمة ونعرضه في الأداة بمفتاح مستقل _widget
            val_key = f"{prefix}_id_in"
            if val_key not in st.session_state:
                st.session_state[val_key] = ""
            
            pdb_input = st.text_input(
                "كود PDB", 
                value=st.session_state[val_key], 
                key=f"{prefix}_id_in_widget"
            ).strip().upper()
            
            st.session_state[val_key] = pdb_input

            if st.button(f"تحميل {p['label']}", key=f"btn_{prefix}"):
                with st.spinner('جاري التحميل...'):
                    # ── الميزة الذكية الذاتية ──
                    # عند إدخال بروتين مصاب، نقوم بالبحث في قاعدة بيانات الطفرات لجلب السليم تلقائياً
                    if prefix == 'm':
                        mdb = load_mutation_db()
                        if pdb_input in mdb:
                            h_id   = mdb[pdb_input]
                            h_data = fetch_pdb_data(h_id)
                            if h_data:
                                # نقوم بتحديث الـ session_state للبروتين السليم برمجياً
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
    شرح التابع:
    يبني شريط التحكم الجانبي (Sidebar) الذي يحتوي على إعدادات العرض البصري ثلاثي الأبعاد
    ومعايير التحليل الهيكلي والمحاذاة.
    
    المخرجات:
    - قاموس يحتوي على كافة خيارات وإعدادات العرض الحالية.
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
    شرح التابع:
    يقوم بتنسيق وعرض:
    1. المشهد التفاعلي ثلاثي الأبعاد للبروتين.
    2. بطاقات الإحصائيات الحيوية (عدد السلاسل، إجمالي الأحماض الأمينية).
    3. ميزة تحميل ملف تسلسل الأحماض بصيغة FASTA.
    4. إجراء التحليل الفراغي الكامل وحساب الجيران لكل حمض في السلسلة وعرض النتائج في جدول تفاعلي.
    """
    prefix = p['prefix']
    colors = PROTEIN_COLORS[prefix]

    # قائمة منسدلة تفاعلية للتركيز البصري والتقريب (Zoom) على طفرة معينة بالبنية
    focus_mut = None
    if highlight:
        mut_opts = ["الكل"] + [f"Residue {m['resi']} (Chain {m['chain']})" for m in highlight]
        sel_mut  = st.selectbox(f"🔍 التركيز على طفرة - {p['label']}", mut_opts, key=f"focus_{prefix}")
        if sel_mut != "الكل":
            parts     = sel_mut.split(" ")
            focus_mut = {'resi': parts[1], 'chain': parts[3].replace(")", "")}

    # حقن الكود التفاعلي لـ py3Dmol في الواجهة
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

    # عرض إحصائيات سريعة ومقاييس البروتين
    all_chains  = get_all_chains(struct)
    total_res   = sum(len([r for r in struct[0][c] if r.id[0] == ' ']) for c in all_chains)
    c1, c2, c3 = st.columns(3)
    c1.metric("عدد السلاسل",   len(all_chains))
    c2.metric("إجمالي الأحماض", total_res)
    c3.metric("السلسلة الحالية", selected_chain)

    # توليد ملف FASTA وإظهار زر التحميل للمستخدم
    fasta = sequence_to_fasta(struct, selected_chain, st.session_state.get(f"{prefix}_id", p['label']))
    if fasta:
        with st.expander(f"🧬 تنزيل FASTA - {p['label']}"):
            st.download_button(
                "⬇️ تنزيل FASTA", fasta,
                f"{st.session_state.get(f'{prefix}_id', 'protein')}_{selected_chain}.fasta",
                "text/plain", key=f"dl_f_{prefix}"
            )

    # زر إطلاق التحليل التفصيلي لحساب المسافات وخصائص الأحماض
    st.divider()
    if st.button(f"🔬 تحليل كامل لـ {p['label']}", key=f"analyze_btn_{prefix}", type="primary"):
        with st.spinner("جاري التحليل..."):
            results = calculate_all_distances(
                f"{st.session_state.get(f'{prefix}_id')}_{selected_chain}",
                pdb_data, selected_chain, radius=settings['search_radius']
            )
            st.session_state[f"{prefix}_results"] = results

    # عرض جدول تفصيلي بالنتائج الهيكلية للأحماض الأمينية عند اكتمال الحساب
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
    شرح التابع:
    يبني ويعرض القسم المشترك لمقارنة السلسلتين (السليم ضد المصاب) ويحتوي على:
    1. جدول المقارنة التفصيلي (الذي يستدعي analyze_impact لتقدير خطورة الطفرة).
    2. رسم بياني تفاعلي متقدم يقارن قيم SASA عبر كامل السلسلة مع وضع علامة مميزة على الطفرات.
    3. لوحة إحصائيات المحاذاة (درجة التطابق Identity % وسكور المحاذاة).
    """
    st.divider()
    st.header("📋 مقارنة السلسلة (Comparison)")

    # استخراج وحساب قيم SASA لكلا البروتينين
    healthy_sasa_map = calculate_sasa_map(structures['h'], h_chain)
    mutant_sasa_map  = calculate_sasa_map(structures['m'], m_chain)
    
    # بناء صفوف جدول المقارنة والتحليل
    rows             = build_comparison_rows(alignment_data, healthy_sasa_map, mutant_sasa_map)
    comparison_df    = pd.DataFrame(rows)

    # 1. عرض جدول المقارنة المتقدم مع تلوين أسطر الطفرات بشكل مميز
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

    # 2. رسم المخطط التفاعلي لمقارنة قيم SASA
    _render_sasa_chart(comparison_df)

    # 3. عرض مقاييس جودة المحاذاة ونسبة التطابق
    _render_alignment_metrics(alignment_data)


def _render_sasa_chart(comparison_df: pd.DataFrame):
    """
    شرح التابع:
    يرسم مخططاً بيانياً تفاعلياً متطوراً لمقارنة منحنى مساحة السطح المعرضة للمذيب (SASA) 
    بين البروتينين، ويقوم بوضع نجوم صفراء مميزة فوق النقاط التي طرأت عليها طفرة.
    """
    st.subheader("📊 مقارنة SASA المتقدمة")
    fig = go.Figure()

    # منحنى البروتين السليم (WT) بلون أخضر نيون متوهج
    fig.add_trace(go.Scatter(
        x=comparison_df['res_num'], y=comparison_df['SASA_H'],
        name='البروتين السليم (WT)',
        line=dict(color='#00ff88', width=2)
    ))
    
    # منحنى البروتين المصاب (MT) بلون أحمر مع ملء شبه شفاف للمنطقة لتوضيح الاختلاف
    fig.add_trace(go.Scatter(
        x=comparison_df['res_num'], y=comparison_df['SASA_M'],
        name='البروتين المصاب (MT)',
        line=dict(color='#ff3333', width=2),
        fill='tonexty', fillcolor='rgba(255, 51, 51, 0.1)'
    ))

    # فلترة وتحديد مواقع الطفرات فقط لوضع نجوم تمثيلية تفاعلية عليها
    mutations_df = comparison_df[comparison_df['السليم'] != comparison_df['المصاب']]
    if not mutations_df.empty:
        fig.add_trace(go.Scatter(
            x=mutations_df['res_num'], y=mutations_df['SASA_M'],
            mode='markers', name='مواقع الطفرات',
            marker=dict(color='yellow', size=10, symbol='star', line=dict(color='black', width=1)),
            hovertemplate="رقم الحمض: %{x}<br>من: %{customdata[0]}<br>إلى: %{customdata[1]}<br>التأثير: %{customdata[2]}<extra></extra>",
            customdata=mutations_df[['السليم', 'المصاب', 'Impact']].values
        ))

    # ضبط خيارات وتنسيقات المظهر الداكن والتفاعلي للرسم البياني
    fig.update_layout(
        template="plotly_dark", height=450, hovermode="x unified",
        xaxis=dict(title="رقم الحمض الأميني", rangeslider=dict(visible=True)),
        yaxis=dict(title="SASA (Å²)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_alignment_metrics(alignment_data: dict):
    """
    شرح التابع:
    يعرض مقاييس ونواتج عملية المحاذاة التسلسلية (تطابق الهوية Identity %، سكور المحاذاة، وفرق الطول).
    كما يظهر لوحة تحتوي على المحاذاة النصية الكاملة المرمزة.
    """
    st.header(f"🧬 Alignment ({alignment_data.get('mode', '').capitalize()})")

    aligned_h = alignment_data['aligned_healthy']
    aligned_m = alignment_data['aligned_mutant']
    
    # حساب نسبة تطابق الأحماض الحقيقية دون احتساب الفجوات المضافة
    identity  = sum(a == b and a != '-' for a, b in zip(aligned_h, aligned_m)) / len(aligned_h) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Alignment Score", round(alignment_data['score'], 1))
    c2.metric("Identity %",      f"{identity:.1f}%")
    c3.metric("فرق الطول",       abs(len(alignment_data['healthy_str']) - len(alignment_data['mutant_str'])))
    
    # عرض شكل المحاذاة النصي المرمز
    st.code(alignment_data['text'], language='text')


# =============================================================================
# نقطة انطلاق التطبيق (App Entry Point)
# =============================================================================

def main():
    """
    الدالة الرئيسية (Main):
    تُنسّق وتحكم تدفق عمل التطبيق بالكامل من البداية وحتى النهاية:
    1. تهيئة الصفحة والعنوان وذاكرة الجلسات.
    2. استدعاء شريط التحكم الجانبي.
    3. بناء لوحتي التحميل للبروتينين (السليم والمصاب) بالتوازي.
    4. معالجة هياكل البروتينات واستخراج السلاسل الببتيدية.
    5. حساب المحاذاة وتحديد الطفرات بشكل متبادل.
    6. رسم الواجهات ثلاثية الأبعاد وتقديم خيارات التحميل والتحليل المنفصل.
    7. استدعاء قسم المقارنة المشترك وعرض الرسوم البيانية والجداول الإحصائية.
    """
    st.set_page_config(page_title="Bio-Impact Analyzer", page_icon="🧬", layout="wide")
    initialize_session_state()
    st.title("🧬 Bio-Impact Analyzer")

    # ── 1. بناء شريط الإعدادات الجانبي ──
    settings = render_sidebar()

    # ── 2. تقسيم الصفحة وعرض لوحتي الإدخال للبروتينين ──
    col1, col2 = st.columns(2)
    proteins = [
        {"label": "السليم",  "prefix": "h", "col": col1},
        {"label": "المصاب",  "prefix": "m", "col": col2},
    ]

    for p in proteins:
        protein_ui_panel(p)

    st.divider()

    # ── 3. معالجة الملفات المرفوعة أو المستدعاة واختيار السلاسل ──
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
                    st.warning("لم يتم العثور على سلاسل ببتيدية في هذا الهيكل.")

    # ── 4. حساب المحاذاة واستخراج الطفرات عند توفر السلسلتين معاً ──
    h_chain      = st.session_state.get('h_selected_chain')
    m_chain      = st.session_state.get('m_selected_chain')
    alignment_data  = None
    highlight_map   = {'h': None, 'm': None}

    if structures.get('h') and structures.get('m') and h_chain and m_chain:
        healthy_seq = get_protein_sequence(structures['h'], h_chain)
        mutant_seq  = get_protein_sequence(structures['m'], m_chain)

        if healthy_seq and mutant_seq:
            # تحويل كلا التسلسلين لحروف مفردة لإجراء المحاذاة
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

            # تحديد المواضع الطافرة لتمييزها وتلوينها لاحقاً
            mut_healthy, mut_mutant = detect_mutations(alignment_data, h_chain, m_chain)
            highlight_map['h'] = mut_healthy or None
            highlight_map['m'] = mut_mutant  or None

    # ── 5. رندر المشاهد ثلاثية الأبعاد وبطاقات الإحصائيات لكل بروتين ──
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

    # ── 6. إظهار قسم المقارنة المشترك والتحليل المتقدم عند توفر البيانات ──
    if structures.get('h') and structures.get('m') and h_chain and m_chain and alignment_data:
        render_comparison_section(structures, h_chain, m_chain, alignment_data)


if __name__ == "__main__":
    main()

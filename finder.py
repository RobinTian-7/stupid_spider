#!/usr/bin/env python3
"""
CSRankings AI Advisor Finder

从 CSRankings 获取目标学校 AI 方向教授列表，爬取主页及 PDF 简历，评分排序。

评分规则（AI/CS 方向专项）:
  招生意向:
    主页明确表示招收本科实习生   +10
    有结构化申请表单 (Google Form) +4
    有泛 opening 信息             +3
  NJU 关联:
    教授本人有南京大学背景         +8
    学生中有南京大学背景           +5
  职称与机会:
    是 Assistant Professor        +5
    新晋 AP (≤3 年入职)           +3  （额外加分）
  经费与算力:
    获得国家级重大基金 (NSF CAREER/ERC/Sloan 等) +4
    获得工业界专项资助 (Google/Amazon/Meta/Nvidia 等) +3
    主页展示 GPU 集群配置          +2
  研究热度:
    研究紧跟前沿 (LLM/多模态/AI4Science 等) +3
  减分项:
    与大厂双重挂靠 (难以回复)      -5
    明确声明不回邮件               -6
    IEEE/ACM/AAAI Fellow          -2  （额外惩罚，代表"功成名就"）
    资深教授 (Full/Chair/Dean)    -3  （明确招本科实习则不扣）

用法:
  python finder.py                              # 交互选方向 → 爬取 → 评分
  python finder.py --areas ML NLP               # 命令行指定方向
  python finder.py --keywords "neuroscience"    # 主页关键词过滤
  python finder.py --dry-run                    # 仅列出教授不爬取
  python finder.py --schools hk sg              # 仅港三新二
  python finder.py --no-cache                   # 忽略缓存重新爬取
  python finder.py --clear-cache                # 清空缓存
"""

import argparse
import csv
import hashlib
import io
import logging
import os
import re
import shutil
import signal
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# ============================================================
# 目标学校
# ============================================================

HONG_KONG_TOP3 = {
    "HKUST": "HKUST",
    "Chinese University of Hong Kong": "CUHK",
    "University of Hong Kong": "HKU",
}
SINGAPORE_TOP2 = {
    "National University of Singapore": "NUS",
    "Nanyang Technological University": "NTU",
}
CANADA = {"University of Toronto": "UofT"}
US_TOP30 = {
    "Massachusetts Inst. of Technology": "MIT",
    "Stanford University": "Stanford",
    "Carnegie Mellon University": "CMU",
    "Univ. of California - Berkeley": "UCB",
    "Univ. of Illinois at Urbana-Champaign": "UIUC",
    "Cornell University": "Cornell",
    "University of Washington": "UW",
    "Georgia Institute of Technology": "GaTech",
    "Princeton University": "Princeton",
    "University of Texas at Austin": "UT-Austin",
    "Univ. of California - Los Angeles": "UCLA",
    "University of Michigan": "UMich",
    "Columbia University": "Columbia",
    "University of Wisconsin - Madison": "UW-Madison",
    "Univ. of California - San Diego": "UCSD",
    "Univ. of Maryland - College Park": "UMD",
    "University of Pennsylvania": "UPenn",
    "Harvard University": "Harvard",
    "New York University": "NYU",
    "Purdue University": "Purdue",
    "Duke University": "Duke",
    "Northwestern University": "Northwestern",
    "University of Southern California": "USC",
    "Rice University": "Rice",
    "Brown University": "Brown",
    "Yale University": "Yale",
    "Johns Hopkins University": "JHU",
    "Univ. of Massachusetts Amherst": "UMass",
    "Ohio State University": "OSU",
    "Univ. of California - Irvine": "UCI",
    "Virginia Tech": "VT",
    "Univ. of California - Santa Barbara": "UCSB",
    "Stony Brook University": "StonyBrook",
    "Rutgers University": "Rutgers",
    "University of Minnesota": "UMN",
}
SCHOOL_GROUPS = {"hk": HONG_KONG_TOP3, "sg": SINGAPORE_TOP2, "ca": CANADA, "us": US_TOP30}

# ============================================================
# AI 研究方向
# ============================================================

AI_AREAS = {
    "AI":       ["ai", "aaai", "ijcai"],
    "Vision":   ["vision", "cvpr", "eccv", "iccv"],
    "ML":       ["mlmining", "icml", "kdd", "iclr", "nips"],
    "NLP":      ["nlp", "acl", "emnlp", "naacl"],
    "Web+IR":   ["inforet", "sigir", "www"],
    "Robotics": ["robotics", "icra", "iros", "rss"],
}
VENUE_TO_AREA = {}
for _a, _vs in AI_AREAS.items():
    for _v in _vs:
        VENUE_TO_AREA[_v] = _a
ALL_AREA_NAMES = list(AI_AREAS.keys())

# ============================================================
# 关键词模式
# ============================================================

UNDERGRAD_INTERN_PATTERNS = [
    r"undergraduate\s+(intern|research\s*assistant|RA|researcher)",
    r"undergrad\s+(intern|research)",
    r"本科.{0,6}(实习|科研|研究助理)",
    r"(looking|searching)\s+for\s+.{0,40}undergraduate",
    r"intern.{0,30}undergraduate",
    r"招收.{0,10}本科",
    r"(hiring|recruiting)\s+.{0,30}undergraduate\s+(student|intern|RA)",
    r"undergraduate\s+student.{0,20}(position|opening|opportunit)",
    r"visiting\s+undergraduate",
    r"(summer|semester)\s+(intern|research).{0,30}undergraduate",
    r"research\s+intern.{0,20}(undergrad|bachelor)",
]
GENERAL_OPENING_PATTERNS = [
    r"(join|joining)\s+(my|our|the)\s+(group|lab|team)",
    r"(position|opening).{0,30}available",
    r"(we|I)\s+(are|am)\s+(hiring|recruiting|looking\s+for)",
    r"prospective\s+student",
    r"looking\s+for\s+.{0,40}(student|researcher|intern|RA)",
    r"(research\s+)?(intern|assistant)\s+(position|opening)",
    r"招生", r"招收", r"open\s+position",
]
NJU_PATTERNS = [r"Nanjing\s+University", r"南京大学", r"\bNJU\b"]
AP_PATTERNS = [r"Assistant\s+Professor", r"助理教授"]
SENIOR_PATTERNS = [
    r"(?<!Associate\s)(?<!Assistant\s)Full\s+Professor",
    r"Chair\s+Professor", r"Chaired\s+Professor",
    r"Distinguished\s+Professor", r"Endowed\s+Professor",
    r"University\s+Professor",
    r"\bDean\b", r"\bAssociate\s+Dean\b",
    r"Department\s+(Head|Chair)", r"Director",
    r"讲座教授", r"(副)?院长", r"系主任",
]
SUBPAGE_LINK_PATTERNS = [
    r"\bjoin\b", r"opening", r"position", r"prospective",
    r"recruit", r"hiring", r"vacanc", r"opportunit",
    r"\bpeople\b", r"\bmember", r"\bstudent", r"\bgroup\b",
    r"\bteam\b", r"\blab\b", r"\badvisee", r"\bmentee", r"\balumni",
    r"\babout\b", r"\bbio\b", r"background", r"education",
]
CV_PDF_PATTERNS = [
    r"\bcv\b", r"resume", r"curriculum[\s\-]*vitae?", r"简历", r"bio\.pdf",
]

# ── AI/CS 专项评分模式 ──────────────────────────────────────

# 大厂双重挂靠（减分）：既是教授又是 Google/Meta/OpenAI 等在职研究员
INDUSTRY_AFFILIATION_PATTERNS = [
    r"Google\s+(Brain|DeepMind|Research)",
    r"Meta\s+(FAIR|AI\s+Research|Research)",
    r"\bOpenAI\b",
    r"Microsoft\s+(Research|MSR)\b",
    r"\bDeepMind\b",
    r"(Research\s+)?(Scientist|Director|Principal\s+Researcher)\s+at\s+"
    r"(Google|Meta|OpenAI|Microsoft|Apple|Amazon|ByteDance|Baidu|Tencent)",
    r"(Staff|Senior|Principal)\s+Research\s+Scientist\s+at",
]

# 国家级重大科研基金（加分）
MAJOR_GRANT_PATTERNS = [
    r"NSF\s+CAREER(\s+Award)?",
    r"\bCAREER\s+Award\b",
    r"ERC\s+(Starting|Consolidator|Advanced)\s+Grant",
    r"Sloan\s+(Research\s+)?Fellowship",
    r"ONR\s+Young\s+Investigator",
    r"DARPA\s+(Young\s+Faculty|Director.?s?\s+Fellowship)",
    r"ARO\s+Young\s+Investigator",
    r"Packard\s+Fellowship",
    r"Simons\s+(Investigator|Fellowship)",
]

# 工业界专项资助（加分）
INDUSTRY_FUNDING_PATTERNS = [
    r"Google\s+Research\s+(Award|Grant|Faculty\s+Award)",
    r"Amazon\s+Research\s+(Award|Grant|Faculty\s+Award)",
    r"Meta\s+Research\s+(Award|Grant)",
    r"Microsoft\s+Research\s+(Award|Grant)",
    r"Nvidia\s+(Academic|Research|Hardware|Faculty)\s+(Grant|Award|Program|Fellowship)",
    r"Adobe\s+Research\s+(Grant|Award)",
    r"Qualcomm\s+(Innovation\s+)?Fellowship",
    r"Samsung\s+Research\s+(Award|Grant)",
]

# GPU 集群 / 算力资源（加分）
GPU_CLUSTER_PATTERNS = [
    r"\b(A100|H100|H200|B200|V100)\b",
    r"GPU\s+(cluster|server|node|farm)",
    r"\d+\s*[×x]\s*(A100|H100|H200|V100)",
    r"(compute|computing)\s+cluster",
    r"(NVIDIA|CUDA)\s+(compute|resource|cluster)",
    r"\d+\s*GPUs?\b",
]

# 前沿研究方向（加分）
HOT_TOPIC_PATTERNS = [
    r"large\s+language\s+model",
    r"\bLLMs?\b",
    r"foundation\s+model",
    r"multi[\s\-]?modal",
    r"AI\s+for\s+(science|health|biology|medicine|drug\s+discovery|climate)",
    r"diffusion\s+model",
    r"generative\s+(AI|model)",
    r"vision[\s\-]language\s+model",
    r"\bRLHF\b",
    r"reinforcement\s+learning\s+from\s+human\s+feedback",
    r"AI\s+agents?\b",
    r"reasoning\s+(in\s+)?(LLM|model|AI)",
    r"in[\s\-]?context\s+learning",
    r"(instruction|alignment)\s+(tuning|training|following)",
]

# 明确声明不回邮件（减分）
NO_EMAIL_PATTERNS = [
    r"cannot\s+reply\s+to\s+(all\s+)?prospective\s+student",
    r"do\s+not\s+(email|contact)\s+me\s+(if|about|regarding|for)",
    r"not\s+able\s+to\s+respond\s+to\s+(all\s+)?email",
    r"please\s+do\s+not\s+(email|send)",
    r"I\s+(cannot|can'?t)\s+respond\s+to\s+(all\s+)?inquir",
    r"not\s+accepting\s+(any\s+)?(more\s+)?(student|intern|applicant)",
    r"not\s+taking\s+(any\s+)?(more\s+)?(student|intern)",
    r"mention\s+my\s+name\s+in\s+your\s+application",
]

# 结构化申请表单（加分）
FORM_RECRUITMENT_PATTERNS = [
    r"forms\.google\.com",
    r"google\.com/forms",
    r"fill\s+(out|in)\s+(this\s+)?(form|survey)",
    r"interest\s+form",
    r"typeform\.com",
    r"airtable\.com",
]

# 新晋 AP（近 ≤3 年入职，加分）—— 当前年份 2026
_NEW_AP_YEAR_PAT = r"20(2[3-9]|3\d)"
NEW_AP_PATTERNS = [
    rf"(joined|joining|started|starting|will\s+join|beginning)\s+.{{0,40}}{_NEW_AP_YEAR_PAT}",
    rf"{_NEW_AP_YEAR_PAT}.{{0,30}}(assistant\s+professor|join|start\s+at|faculty)",
    r"new\s+faculty\s+(member|hire|position)",
    r"recently\s+joined",
    r"incoming\s+(assistant\s+)?faculty",
    r"I\s+(am|will\s+be)\s+(joining|starting)\s+as\s+(an?\s+)?[Aa]ssistant\s+[Pp]rofessor",
]

# IEEE/ACM/AAAI Fellow（额外减分，代表"功成名就"）
FELLOW_PATTERNS = [
    r"\bIEEE\s+Fellow\b",
    r"\bACM\s+Fellow\b",
    r"\bAAAI\s+Fellow\b",
    r"\bNAS\s+(Member|Fellow)\b",
    r"\bNAE\s+(Member|Fellow)\b",
]

CSRANKINGS_CSV_URL = "https://raw.githubusercontent.com/emeryberger/CSrankings/gh-pages/csrankings.csv"
AUTHOR_INFO_URL = "https://raw.githubusercontent.com/emeryberger/CSrankings/gh-pages/generated-author-info.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
}

# ============================================================
# 数据结构
# ============================================================

def clean_name(name):
    """去掉 CSRankings 的消歧后缀，如 ' 0001'。"""
    return re.sub(r"\s+\d{4}$", "", name)


@dataclass
class Professor:
    name: str
    affiliation: str
    homepage: str
    scholar_id: str
    school_abbr: str = ""
    areas: list = field(default_factory=list)
    score: int = 0
    # 招生信号
    has_undergrad_intern: bool = False
    has_general_opening: bool = False
    has_form_recruitment: bool = False
    # NJU 关联
    has_nju_background: bool = False
    has_nju_students: bool = False
    # 职称
    is_ap: bool = False
    is_new_ap: bool = False
    is_senior: bool = False
    is_fellow: bool = False
    # 资源与经费
    has_major_grant: bool = False
    has_industry_funding: bool = False
    has_gpu_cluster: bool = False
    # 研究热度
    is_hot_topic: bool = False
    # 风险标记
    has_industry_affiliation: bool = False
    has_no_email_policy: bool = False
    # 爬取元数据
    cv_found: bool = False
    keywords_matched: bool = False
    pages_scraped: int = 0
    intern_evidence: list = field(default_factory=list)
    nju_evidence: list = field(default_factory=list)
    error: str = ""

    @property
    def display_name(self):
        return clean_name(self.name)

    def reset_analysis(self):
        """重置分析结果，用于 Playwright 重试前清空状态，防止分数累加。"""
        self.score = 0
        self.has_undergrad_intern = False
        self.has_general_opening = False
        self.has_form_recruitment = False
        self.has_nju_background = False
        self.has_nju_students = False
        self.is_ap = False
        self.is_new_ap = False
        self.is_senior = False
        self.is_fellow = False
        self.has_major_grant = False
        self.has_industry_funding = False
        self.has_gpu_cluster = False
        self.is_hot_topic = False
        self.has_industry_affiliation = False
        self.has_no_email_policy = False
        self.cv_found = False
        self.keywords_matched = False
        self.pages_scraped = 0
        self.intern_evidence = []
        self.nju_evidence = []
        self.error = ""


# ============================================================
# 缓存
# ============================================================

class Cache:
    def __init__(self, cache_dir="cache", page_ttl=7 * 86400, data_ttl=86400):
        self.cache_dir = cache_dir
        self.pages_dir = os.path.join(cache_dir, "pages")
        self.page_ttl = page_ttl
        self.data_ttl = data_ttl
        os.makedirs(self.pages_dir, exist_ok=True)

    def _path(self, key):
        return os.path.join(self.pages_dir, hashlib.md5(key.encode()).hexdigest())

    def get_page(self, key):
        p = self._path(key)
        if not os.path.exists(p):
            return None
        if time.time() - os.path.getmtime(p) > self.page_ttl:
            return None
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return None

    def set_page(self, key, content):
        try:
            with open(self._path(key), "w", encoding="utf-8", errors="replace") as f:
                f.write(content)
        except Exception:
            pass

    def get_data(self, name):
        p = os.path.join(self.cache_dir, name)
        if not os.path.exists(p):
            return None
        if time.time() - os.path.getmtime(p) > self.data_ttl:
            return None
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    def set_data(self, name, content):
        with open(os.path.join(self.cache_dir, name), "w", encoding="utf-8") as f:
            f.write(content)

    def clear(self):
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.pages_dir, exist_ok=True)

    @property
    def page_count(self):
        try:
            return len(os.listdir(self.pages_dir))
        except Exception:
            return 0


class NoCache:
    def get_page(self, _):   return None
    def set_page(self, *_):  pass
    def get_data(self, _):   return None
    def set_data(self, *_):  pass
    def clear(self):         pass
    @property
    def page_count(self):    return 0


# ============================================================
# 增量 CSV 写入（线程安全）
# ============================================================

class IncrementalCSV:
    """每分析完一位教授立即写入 CSV，中断时不丢失已有结果。"""

    HEADER = [
        "分数", "学校", "姓名", "院校全称", "研究方向", "主页", "Google Scholar",
        # 招生信号
        "明确招本科实习", "有申请表单", "有opening",
        # NJU
        "教授NJU背景", "学生NJU背景",
        # 职称
        "是否AP", "新晋AP(≤3年)", "资深教授", "是否Fellow",
        # 资源
        "国家级基金", "工业界资助", "有GPU集群",
        # 研究热度
        "前沿方向(LLM等)",
        # 风险
        "大厂双重挂靠", "明确不回邮件",
        # 元数据
        "找到CV", "关键词匹配", "爬取页数",
        "招生证据", "NJU证据", "错误",
    ]

    def __init__(self, filepath):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.f = open(filepath, "w", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.f)
        self.writer.writerow(self.HEADER)
        self.f.flush()

    def write(self, p):
        scholar = (
            f"https://scholar.google.com/citations?user={p.scholar_id}"
            if p.scholar_id else ""
        )
        row = [
            p.score, p.school_abbr, p.display_name, p.affiliation,
            ", ".join(p.areas), p.homepage, scholar,
            # 招生信号
            p.has_undergrad_intern, p.has_form_recruitment, p.has_general_opening,
            # NJU
            p.has_nju_background, p.has_nju_students,
            # 职称
            p.is_ap, p.is_new_ap, p.is_senior, p.is_fellow,
            # 资源
            p.has_major_grant, p.has_industry_funding, p.has_gpu_cluster,
            # 热度
            p.is_hot_topic,
            # 风险
            p.has_industry_affiliation, p.has_no_email_policy,
            # 元数据
            p.cv_found, p.keywords_matched, p.pages_scraped,
            " | ".join(p.intern_evidence[:2]),
            " | ".join(p.nju_evidence[:2]),
            p.error,
        ]
        with self.lock:
            self.writer.writerow(row)
            self.f.flush()

    def close(self):
        self.f.close()


# ============================================================
# HTTP 请求（带重试）
# ============================================================

def request_with_retry(session, url, timeout=12, max_retries=1):
    """GET 请求，失败自动重试 1 次（间隔 2 秒）。"""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2)
    raise last_err


# ============================================================
# 方向选择
# ============================================================

def select_areas_interactive():
    print("\n" + "=" * 50)
    print("请选择 AI 子方向（可多选）")
    print("=" * 50)
    for i, name in enumerate(ALL_AREA_NAMES):
        venues = ", ".join(AI_AREAS[name])
        print(f"  {i + 1}. {name:10s} ({venues})")
    print(f"  0. 全部")
    print()
    while True:
        raw = input("输入编号（如 1,3 或直接回车=全部）: ").strip()
        if not raw:
            return None
        parts = re.split(r"[,\s]+", raw)
        try:
            nums = [int(x) for x in parts]
        except ValueError:
            print("请输入数字。")
            continue
        if 0 in nums:
            return None
        selected = []
        for n in nums:
            if 1 <= n <= len(ALL_AREA_NAMES):
                selected.append(ALL_AREA_NAMES[n - 1])
            else:
                print(f"编号 {n} 无效")
                selected = []
                break
        if selected:
            print(f"已选：{', '.join(selected)}")
            return list(dict.fromkeys(selected))


# ============================================================
# 数据获取
# ============================================================

def build_target_schools(groups):
    schools = {}
    for g in groups:
        if g in SCHOOL_GROUPS:
            schools.update(SCHOOL_GROUPS[g])
    return schools


def match_school(affiliation, target_schools):
    aff_lower = affiliation.lower().strip()
    for name, abbr in target_schools.items():
        if name.lower() == aff_lower:
            return abbr
    candidates = []
    for name, abbr in target_schools.items():
        nl = name.lower()
        if nl in aff_lower or aff_lower in nl:
            candidates.append((len(name), abbr))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


def fetch_csv(url, cache, cache_name):
    cached = cache.get_data(cache_name)
    if cached is not None:
        logging.info(f"使用缓存: {cache_name}")
        return cached
    logging.info(f"正在下载 {cache_name}...")
    resp = requests.get(url, timeout=60, headers=HEADERS)
    resp.raise_for_status()
    cache.set_data(cache_name, resp.text)
    return resp.text


def fetch_author_areas(target_schools, cache):
    text = fetch_csv(AUTHOR_INFO_URL, cache, "author_info.csv")
    reader = csv.DictReader(io.StringIO(text))
    author_areas = defaultdict(set)
    for row in reader:
        name = row.get("name", "").strip()
        dept = row.get("dept", "").strip()
        venue = row.get("area", "").strip()
        area_name = VENUE_TO_AREA.get(venue)
        if not name or not area_name:
            continue
        abbr = match_school(dept, target_schools)
        if abbr:
            author_areas[(name.lower(), abbr)].add(area_name)
    logging.info(f"已获取 {len(author_areas)} 位教授的 AI 方向信息")
    return author_areas


def fetch_professors(target_schools, author_areas, selected_areas, cache):
    text = fetch_csv(CSRANKINGS_CSV_URL, cache, "csrankings.csv")
    reader = csv.DictReader(io.StringIO(text))
    professors = []
    seen = set()
    selected_set = set(selected_areas) if selected_areas else None

    for row in reader:
        name = row.get("name", "").strip()
        affiliation = row.get("affiliation", "").strip()
        homepage = row.get("homepage", "").strip()
        scholar_id = row.get("scholarid", "").strip()
        if not name or not affiliation:
            continue
        abbr = match_school(affiliation, target_schools)
        if not abbr:
            continue
        key = (name.lower(), abbr)
        if key in seen:
            continue
        seen.add(key)

        prof_areas = sorted(author_areas.get(key, set()))
        if not prof_areas:
            continue
        if selected_set and not (set(prof_areas) & selected_set):
            continue

        professors.append(Professor(
            name=name, affiliation=affiliation, homepage=homepage,
            scholar_id=scholar_id, school_abbr=abbr, areas=prof_areas,
        ))
    return professors


# ============================================================
# 页面爬取
# ============================================================

def fetch_html(url, session, cache, timeout=12):
    cached = cache.get_page(url)
    if cached is not None:
        return cached, True
    try:
        resp = request_with_retry(session, url, timeout=timeout)
        cache.set_page(url, resp.text)
        return resp.text, False
    except Exception:
        return None, False


def parse_html(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True), soup


def extract_pdf_text(pdf_bytes, max_pages=20):
    if not HAS_PYMUPDF:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = [doc[i].get_text() for i in range(min(len(doc), max_pages))]
        doc.close()
        return "\n".join(parts)
    except Exception:
        return ""


def fetch_pdf_text(url, session, cache, max_size=10 * 1024 * 1024):
    cache_key = f"pdf:{url}"
    cached = cache.get_page(cache_key)
    if cached is not None:
        return cached
    if not HAS_PYMUPDF:
        return ""
    try:
        resp = request_with_retry(session, url, timeout=20)
        if len(resp.content) > max_size:
            return ""
        text = extract_pdf_text(resp.content)
        cache.set_page(cache_key, text)
        return text
    except Exception:
        return ""


def find_subpage_urls(soup, base_url, limit=8):
    if soup is None:
        return []
    urls, seen = [], set()
    base_domain = urlparse(base_url).netloc
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        if urlparse(full).netloc != base_domain:
            continue
        if full in seen or full.rstrip("/") == base_url.rstrip("/"):
            continue
        if re.search(r"\.(pdf|doc|ppt|zip|gz|tar|png|jpg|jpeg|gif|svg|bib|tex)$", href, re.I):
            continue
        combined = f"{a.get_text(strip=True)} {href}".lower()
        for pat in SUBPAGE_LINK_PATTERNS:
            if re.search(pat, combined):
                seen.add(full)
                urls.append(full)
                break
        if len(urls) >= limit:
            break
    return urls


def find_cv_pdf_urls(soup, base_url, limit=3):
    if soup is None:
        return []
    urls, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        combined = f"{a.get_text(strip=True)} {href}".lower()
        for pat in CV_PDF_PATTERNS:
            if re.search(pat, combined):
                full = urljoin(base_url, href)
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
                break
        if len(urls) >= limit:
            break
    return urls


def find_form_urls(soup):
    """检测页面链接中是否存在结构化申请表单。"""
    if soup is None:
        return False
    for a in soup.find_all("a", href=True):
        if re.search(
            r"forms\.google\.com|google\.com/forms|typeform\.com|airtable\.com",
            a["href"], re.I,
        ):
            return True
    return False


# ============================================================
# Playwright fallback（可选，用于 JS 渲染页面）
# ============================================================

def fetch_html_playwright(url, timeout=15000):
    """用无头浏览器获取页面 HTML。仅在 requests 失败时调用。"""
    if not HAS_PLAYWRIGHT:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            html = page.content()
            browser.close()
            return html
    except Exception:
        return None


def playwright_retry_failures(failed_profs, cache):
    """对爬取失败的教授用 Playwright 重试。单线程顺序执行。"""
    if not HAS_PLAYWRIGHT or not failed_profs:
        return []

    logging.info(f"使用 Playwright 重试 {len(failed_profs)} 个失败页面...")
    retried = []
    for prof in failed_profs:
        html = fetch_html_playwright(prof.homepage)
        if html:
            cache.set_page(prof.homepage, html)
            retried.append(prof)
            logging.info(f"  Playwright 成功: {prof.display_name}")
        time.sleep(0.5)
    return retried


# ============================================================
# 分析
# ============================================================

def check_patterns(text, patterns):
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def extract_snippets(text, patterns, context=60, max_snippets=3):
    snippets = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s = max(0, m.start() - context)
            e = min(len(text), m.end() + context)
            snippets.append("..." + re.sub(r"\s+", " ", text[s:e].strip()) + "...")
            if len(snippets) >= max_snippets:
                return snippets
    return snippets


def analyze_nju_context(all_text, main_soup):
    if not check_patterns(all_text, NJU_PATTERNS):
        return False, False
    has_bio, has_student = False, False

    if main_soup is not None:
        bio_text, student_text = "", ""
        for h in main_soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
            ht = h.get_text(strip=True).lower()
            parts = []
            for sib in h.find_next_siblings():
                if sib.name in ["h1", "h2", "h3", "h4", "h5"]:
                    break
                parts.append(sib.get_text(strip=True))
            sec = " ".join(parts)
            if any(k in ht for k in [
                "bio", "about", "education", "background", "cv", "vita",
                "experience", "简历", "教育", "经历",
            ]):
                bio_text += " " + sec
            elif any(k in ht for k in [
                "student", "people", "group", "team", "member",
                "advisee", "mentee", "alumni", "学生", "成员",
            ]):
                student_text += " " + sec
        if bio_text:
            has_bio = check_patterns(bio_text, NJU_PATTERNS)
        if student_text:
            has_student = check_patterns(student_text, NJU_PATTERNS)
        if has_bio or has_student:
            return has_bio, has_student

    # Fallback
    split = len(all_text) * 2 // 5
    has_bio = check_patterns(all_text[:split], NJU_PATTERNS)
    has_student = check_patterns(all_text[split:], NJU_PATTERNS)
    if not has_bio and not has_student:
        has_student = True
    return has_bio, has_student


def parse_education_from_text(text):
    """在 CV/简历文本中定位 Education 段落，检测 NJU。"""
    edu_match = re.search(
        r"(education|学历|教育背景|academic\s+background)"
        r"(.*?)"
        r"(experience|work|research|publication|award|honor|skill|项目|工作|$)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if edu_match:
        return check_patterns(edu_match.group(2), NJU_PATTERNS)
    return False


def analyze_professor(prof, cache, keywords=None):
    """爬取并分析一位教授的全部在线信息。"""
    if not prof.homepage:
        prof.error = "无主页"
        return prof

    session = requests.Session()
    session.headers.update(HEADERS)

    # --- 主页 ---
    main_html, _ = fetch_html(prof.homepage, session, cache)
    if not main_html:
        prof.error = "主页获取失败"
        return prof

    main_text, main_soup = parse_html(main_html)
    if not main_text:
        prof.error = "主页无内容"
        return prof

    all_text = main_text
    pages = 1

    # --- 子页面 ---
    subpage_urls = find_subpage_urls(main_soup, prof.homepage)
    all_cv_pdfs = find_cv_pdf_urls(main_soup, prof.homepage)

    for sub_url in subpage_urls:
        sub_html, sub_cached = fetch_html(sub_url, session, cache, timeout=8)
        if sub_html:
            sub_text, sub_soup = parse_html(sub_html)
            all_text += " " + sub_text
            pages += 1
            all_cv_pdfs.extend(find_cv_pdf_urls(sub_soup, sub_url))
        if not sub_cached:
            time.sleep(0.15)

    # --- PDF 简历 ---
    seen_pdfs = set()
    for pdf_url in all_cv_pdfs[:5]:
        if pdf_url in seen_pdfs:
            continue
        seen_pdfs.add(pdf_url)
        pdf_text = fetch_pdf_text(pdf_url, session, cache)
        if pdf_text:
            all_text += " " + pdf_text
            prof.cv_found = True
            pages += 1
            if parse_education_from_text(pdf_text):
                prof.has_nju_background = True

    prof.pages_scraped = pages

    # --- 关键词过滤 ---
    if keywords:
        kw_pattern = "|".join(re.escape(k) for k in keywords)
        prof.keywords_matched = bool(re.search(kw_pattern, all_text, re.IGNORECASE))

    # ── 招生信号 ────────────────────────────────────────────
    prof.has_undergrad_intern = check_patterns(all_text, UNDERGRAD_INTERN_PATTERNS)
    if not prof.has_undergrad_intern:
        prof.has_general_opening = check_patterns(all_text, GENERAL_OPENING_PATTERNS)
    # 结构化表单：文本匹配 + href 检测双保险
    prof.has_form_recruitment = (
        check_patterns(all_text, FORM_RECRUITMENT_PATTERNS)
        or find_form_urls(main_soup)
    )
    # 明确不回邮件
    prof.has_no_email_policy = check_patterns(all_text, NO_EMAIL_PATTERNS)

    # ── NJU（主页补充，CV 中已检测的不覆盖） ────────────────
    bio, stu = analyze_nju_context(all_text, main_soup)
    if bio:
        prof.has_nju_background = True
    if stu:
        prof.has_nju_students = True

    # ── 职称 ─────────────────────────────────────────────────
    prof.is_ap = check_patterns(all_text, AP_PATTERNS)
    prof.is_senior = check_patterns(all_text, SENIOR_PATTERNS)
    prof.is_fellow = check_patterns(all_text, FELLOW_PATTERNS)
    if prof.is_ap:
        prof.is_new_ap = check_patterns(all_text, NEW_AP_PATTERNS)

    # ── 资源与经费 ───────────────────────────────────────────
    prof.has_major_grant = check_patterns(all_text, MAJOR_GRANT_PATTERNS)
    prof.has_industry_funding = check_patterns(all_text, INDUSTRY_FUNDING_PATTERNS)
    prof.has_gpu_cluster = check_patterns(all_text, GPU_CLUSTER_PATTERNS)

    # ── 研究热度 & 大厂挂靠 ──────────────────────────────────
    prof.is_hot_topic = check_patterns(all_text, HOT_TOPIC_PATTERNS)
    prof.has_industry_affiliation = check_patterns(all_text, INDUSTRY_AFFILIATION_PATTERNS)

    # ── 证据片段 ─────────────────────────────────────────────
    prof.intern_evidence = extract_snippets(
        all_text, UNDERGRAD_INTERN_PATTERNS + GENERAL_OPENING_PATTERNS,
    )
    prof.nju_evidence = extract_snippets(all_text, NJU_PATTERNS)

    # ── 评分 ─────────────────────────────────────────────────
    # 招生意向（最高权重）
    if prof.has_undergrad_intern:
        prof.score += 10
    if prof.has_form_recruitment:
        prof.score += 4
    if prof.has_general_opening:
        prof.score += 3

    # NJU 关联
    if prof.has_nju_background:
        prof.score += 8
    if prof.has_nju_students:
        prof.score += 5

    # 职称
    if prof.is_ap:
        prof.score += 5
    if prof.is_new_ap:
        prof.score += 3   # 新晋 AP 额外加分

    # 资源与经费
    if prof.has_major_grant:
        prof.score += 4
    if prof.has_industry_funding:
        prof.score += 3
    if prof.has_gpu_cluster:
        prof.score += 2

    # 研究热度
    if prof.is_hot_topic:
        prof.score += 3

    # 减分项
    if prof.is_senior and not prof.has_undergrad_intern:
        prof.score -= 3
    if prof.is_fellow:
        prof.score -= 2
    if prof.has_industry_affiliation:
        prof.score -= 5
    if prof.has_no_email_policy:
        prof.score -= 6

    return prof


# ============================================================
# 输出
# ============================================================

def print_results(results):
    results.sort(key=lambda p: (-p.score, p.school_abbr, p.name))

    print()
    print("=" * 170)
    print(
        f"{'分数':>4} | {'学校':8} | {'姓名':25} | {'方向':18} | "
        f"{'职称':10} | {'招生':10} | {'NJU':8} | "
        f"{'资源':12} | {'热度':4} | {'风险':12} | 主页"
    )
    print("-" * 170)

    for p in results:
        if p.score <= 0:
            continue

        # 职称
        rank_parts = []
        if p.is_new_ap:
            rank_parts.append("新AP")
        elif p.is_ap:
            rank_parts.append("AP")
        if p.is_fellow:
            rank_parts.append("Fellow")
        elif p.is_senior:
            rank_parts.append("资深")
        rank = "/".join(rank_parts) if rank_parts else ""

        # 招生
        if p.has_undergrad_intern:
            intern_f = "明确招实习"
        elif p.has_form_recruitment:
            intern_f = "有表单"
        elif p.has_general_opening:
            intern_f = "有opening"
        else:
            intern_f = ""

        # NJU
        nju_parts = []
        if p.has_nju_background:
            nju_parts.append("本人")
        if p.has_nju_students:
            nju_parts.append("学生")
        nju_f = "+".join(nju_parts) if nju_parts else ""

        # 资源
        res_parts = []
        if p.has_major_grant:
            res_parts.append("国家奖")
        if p.has_industry_funding:
            res_parts.append("工业资助")
        if p.has_gpu_cluster:
            res_parts.append("GPU")
        res_f = "/".join(res_parts) if res_parts else ""

        # 风险
        risk_parts = []
        if p.has_industry_affiliation:
            risk_parts.append("大厂挂靠")
        if p.has_no_email_policy:
            risk_parts.append("不回邮件")
        risk_f = "/".join(risk_parts) if risk_parts else ""

        areas_s = ",".join(p.areas[:3]) + ("..." if len(p.areas) > 3 else "")
        hot_f = "热" if p.is_hot_topic else ""

        print(
            f"{p.score:4d} | {p.school_abbr:8s} | {p.display_name:25s} | "
            f"{areas_s:18s} | {rank:10s} | {intern_f:10s} | {nju_f:8s} | "
            f"{res_f:12s} | {hot_f:4s} | {risk_f:12s} | {p.homepage}"
        )

    total = len(results)
    scored = sum(1 for p in results if p.score > 0)
    print()
    print(
        f"共 {total} 位 | 得分>0: {scored} | "
        f"招实习: {sum(1 for p in results if p.has_undergrad_intern)} | "
        f"有表单: {sum(1 for p in results if p.has_form_recruitment)} | "
        f"有opening: {sum(1 for p in results if p.has_general_opening)} | "
        f"NJU关联: {sum(1 for p in results if p.has_nju_background or p.has_nju_students)} | "
        f"新AP: {sum(1 for p in results if p.is_new_ap)} | "
        f"AP: {sum(1 for p in results if p.is_ap and not p.is_new_ap)} | "
        f"国家奖: {sum(1 for p in results if p.has_major_grant)} | "
        f"工业资助: {sum(1 for p in results if p.has_industry_funding)} | "
        f"热门方向: {sum(1 for p in results if p.is_hot_topic)} | "
        f"大厂挂靠: {sum(1 for p in results if p.has_industry_affiliation)} | "
        f"不回邮件: {sum(1 for p in results if p.has_no_email_policy)} | "
        f"失败: {sum(1 for p in results if p.error)} | "
        f"共爬 {sum(p.pages_scraped for p in results)} 页"
    )


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CSRankings AI 导师筛选工具")
    parser.add_argument("--dry-run", action="store_true", help="仅列出教授不爬取")
    parser.add_argument("--threads", type=int, default=8, help="并发线程数")
    parser.add_argument("--output", default="advisor_results.csv", help="输出 CSV")
    parser.add_argument("--schools", nargs="*", default=["hk", "sg", "ca", "us"],
                        choices=["hk", "sg", "ca", "us"])
    parser.add_argument("--areas", nargs="*", default=None, metavar="AREA",
                        help=f"AI 子方向: {', '.join(ALL_AREA_NAMES)}")
    parser.add_argument("--keywords", nargs="*", default=None, metavar="KW",
                        help="主页关键词过滤（如 'neuroscience' 'LLM'）")
    parser.add_argument("--no-cache", action="store_true", help="忽略缓存")
    parser.add_argument("--clear-cache", action="store_true", help="清空缓存")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if not HAS_PYMUPDF:
        logging.warning("未安装 PyMuPDF，PDF 简历无法解析。pip install PyMuPDF")
    if HAS_PLAYWRIGHT:
        logging.info("Playwright 可用，将对失败页面自动重试")

    # 缓存
    cache = NoCache() if args.no_cache else Cache()
    if args.clear_cache:
        cache = Cache()
        cache.clear()
        logging.info("缓存已清空")

    # 学校
    target_schools = build_target_schools(args.schools)
    if not target_schools:
        sys.exit("没有选择学校组！")
    logging.info(f"目标学校: {len(target_schools)} 所")

    # 方向
    author_areas = fetch_author_areas(target_schools, cache)
    if args.areas is not None:
        selected = []
        for a in args.areas:
            matched = [n for n in ALL_AREA_NAMES if n.lower() == a.lower()]
            if matched:
                selected.append(matched[0])
            else:
                sys.exit(f"未知方向 '{a}'。可选: {', '.join(ALL_AREA_NAMES)}")
        selected_areas = selected or None
    else:
        selected_areas = select_areas_interactive()

    if selected_areas:
        logging.info(f"已选方向: {', '.join(selected_areas)}")
    if args.keywords:
        logging.info(f"关键词过滤: {', '.join(args.keywords)}")

    # 教授列表
    professors = fetch_professors(target_schools, author_areas, selected_areas, cache)
    logging.info(f"共 {len(professors)} 位教授匹配条件")
    if not professors:
        return

    if args.dry_run:
        by_school = {}
        for p in professors:
            by_school.setdefault(p.school_abbr, []).append(p)
        for abbr in sorted(by_school):
            profs = by_school[abbr]
            print(f"\n--- {abbr} ({len(profs)} 人) ---")
            for p in sorted(profs, key=lambda x: x.name):
                print(f"  {p.display_name:30s} [{', '.join(p.areas):25s}] {p.homepage}")
        print(f"\n共 {len(professors)} 位教授")
        return

    # --- 爬取 ---
    cached_before = cache.page_count
    logging.info(f"开始爬取（{args.threads} 线程，缓存 {cached_before} 页）...")

    csv_writer = IncrementalCSV(args.output)
    results = []
    failed = []
    interrupted = False

    def on_signal(_s, _f):
        nonlocal interrupted
        interrupted = True
        logging.warning("收到中断，正在保存...")

    signal.signal(signal.SIGINT, on_signal)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(analyze_professor, p, cache, args.keywords): p
            for p in professors
        }
        done = 0
        total = len(professors)
        for future in as_completed(futures):
            if interrupted:
                break
            try:
                prof = future.result(timeout=60)
            except Exception as e:
                prof = futures[future]
                prof.error = str(e)
            results.append(prof)
            csv_writer.write(prof)
            if prof.error:
                failed.append(prof)
            done += 1
            if done % 20 == 0 or done == total:
                logging.info(f"进度: {done}/{total}")

    csv_writer.close()

    # --- Playwright 重试（修复：重置状态 + 异常捕获）---
    if HAS_PLAYWRIGHT and failed and not interrupted:
        retried = playwright_retry_failures(failed, cache)
        if retried:
            logging.info(f"对 {len(retried)} 位 Playwright 成功的教授重新分析...")
            for prof in retried:
                prof.reset_analysis()   # 清空旧状态，防止分数累加
                try:
                    analyze_professor(prof, cache, args.keywords)
                except Exception as e:
                    prof.error = f"retry: {e}"
                    logging.warning(f"  重试分析失败 {prof.display_name}: {e}")

    # --- 关键词过滤 ---
    if args.keywords:
        before = len(results)
        results = [p for p in results if p.keywords_matched or p.score > 5]
        logging.info(f"关键词过滤: {before} → {len(results)} 人")

    # --- 最终输出 ---
    print_results(results)

    # 重写完整排序后的 CSV
    csv_final = IncrementalCSV(args.output)
    results.sort(key=lambda p: (-p.score, p.school_abbr, p.name))
    for p in results:
        csv_final.write(p)
    csv_final.close()

    new_pages = cache.page_count - cached_before
    if new_pages > 0:
        logging.info(f"新缓存 {new_pages} 页")
    logging.info(f"结果已保存到 {args.output}")


if __name__ == "__main__":
    main()

import os
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from docx import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import requests
import streamlit as st

# ================= 1. 页面基本配置与高级 UI 美化 =================
st.set_page_config(
    page_title="InsightHub - 智能文档分析平台",
    page_icon="✨",
    layout="wide",
)

# 注入自定义 CSS
st.markdown(
    """
    <style>
    /* 全局背景与字体美化 */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    header {visibility: hidden;}
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    /* 美化按钮 */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%);
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    
    /* 美化选项卡 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ================= 2. 🔐 后端无感获取 API Key (完全隐藏) =================
def init_api_config():
  """安全加载 API Key，优先寻找 Secrets，其次寻找系统环境变量。"""
  api_key = None
  api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

  # 1. 尝试从 Streamlit Secrets (secrets.toml) 读取
  if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
    api_base = st.secrets.get("OPENAI_API_BASE", api_base)
  # 2. 尝试从环境变量读取
  elif os.getenv("OPENAI_API_KEY"):
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", api_base)

  if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    return True
  else:
    st.error(
        "🚨 系统后台未配置 API Key，请联系管理员在 `.streamlit/secrets.toml`"
        " 中添加配置。"
    )
    return False


has_api = init_api_config()

# ================= 3. 数据解析与多线程并发工具 =================


def fetch_article_from_url(url):
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }
  try:
    response = requests.get(url, headers=headers, timeout=10)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join(
        [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
    )
  except Exception as e:
    st.error(f"网页抓取失败: {e}")
    return None


def parse_uploaded_file(uploaded_file):
  file_type = uploaded_file.name.split(".")[-1].lower()
  text = ""
  try:
    if file_type == "pdf":
      reader = PdfReader(uploaded_file)
      for page in reader.pages:
        text += page.extract_text() or ""
    elif file_type in ["doc", "docx"]:
      doc = Document(uploaded_file)
      text = "\n".join([p.text for p in doc.paragraphs if p.text])
    elif file_type == "txt":
      text = uploaded_file.read().decode("utf-8")
    return text
  except Exception as e:
    st.error(f"文件解析失败: {e}")
    return None


def process_chunk_map(chunk, chain):
  try:
    res = chain.invoke({"chunk": chunk})
    return res.content
  except Exception:
    return ""


def run_parallel_map(chunks, chain, max_workers=10):
  results = [None] * len(chunks)
  progress_bar = st.progress(0)
  status_text = st.empty()
  completed_count = 0

  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_index = {
        executor.submit(process_chunk_map, chunk, chain): i
        for i, chunk in enumerate(chunks)
    }
    for future in future_to_index:
      i = future_to_index[future]
      results[i] = future.result()
      completed_count += 1
      progress_bar.progress(completed_count / len(chunks))
      status_text.text(
          f"⚡ 高并发引擎工作中: 已完成 {completed_count}/{len(chunks)} 个片段解析..."
      )

  status_text.text("✨ 切片处理完成，正在整合生成商业报告...")
  return results


# ================= 4. 头部 Header =================
st.markdown(
    '<div class="main-header">InsightHub 智能文档与资讯分析平台</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">企业级多源文档结构化提炼 • 高并发 Map-Reduce 引擎 •'
    " 交互式对话</div>",
    unsafe_allow_html=True,
)

# ================= 5. 输入控制区 =================
tab1, tab2 = st.tabs(
    ["🔗 网页链接解析", "📁 本地文档分析 (支持超长文本)"]
)

raw_text = None

with tab1:
  col_input, col_btn = st.columns([4, 1])
  with col_input:
    target_url = st.text_input(
        "网页 URL",
        placeholder="https://example.com/article",
        label_visibility="collapsed",
    )
  with col_btn:
    if st.button("开始提取", key="url_btn") and has_api:
      if target_url:
        with st.spinner("正在抓取网页..."):
          raw_text = fetch_article_from_url(target_url)

with tab2:
  uploaded_file = st.file_uploader(
      "上传 PDF / Word / TXT 文件",
      type=["pdf", "docx", "txt"],
      label_visibility="collapsed",
  )
  if st.button("开始分析文档", key="file_btn") and has_api:
    if uploaded_file:
      with st.spinner("正在读取文件..."):
        raw_text = parse_uploaded_file(uploaded_file)
    else:
      st.warning("请先选择文件！")

# ================= 6. 分析逻辑触发与 Session 维持 =================
if raw_text:
  st.session_state["raw_text"] = raw_text

  llm = ChatOpenAI(model="qwen-plus", temperature=0.3)

  if len(raw_text) <= 3000:
    with st.spinner("AI 正在提炼报告..."):
      prompt = PromptTemplate.from_template(
          """你是一名资深商业分析师。请对以下内容深入分析并按格式输出：
            1. 核心主题（15字以内）
            2. 核心事实/要点总结（3-5条，带粗体小标题）
            3. 关键结论/战略建议（2条）
            4. 标签/关键词（3-5个，逗号分隔）

            内容：{content}
            """
      )
      chain = prompt | llm
      st.session_state["final_report"] = chain.invoke(
          {"content": raw_text}
      ).content
  else:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200
    )
    chunks = text_splitter.split_text(raw_text)

    map_prompt = PromptTemplate.from_template(
        "请简要总结以下文本片段的核心要点，用列表列出：\n\n{chunk}"
    )
    map_chain = map_prompt | llm

    intermediate_summaries = run_parallel_map(
        chunks, map_chain, max_workers=10
    )
    combined_summaries = "\n\n".join([s for s in intermediate_summaries if s])

    reduce_prompt = PromptTemplate.from_template(
        """你是一名资深首席分析师。以下是对长文档各个章节的局部总结：

            {combined_summaries}

            请结合上述所有局部总结，整合输出一份高价值的商业分析报告：
            1. 整体核心主题（15字以内）
            2. 核心事实/要点总结（4-6条，按重要性排序，带粗体小标题）
            3. 综合结论/战略建议（3条）
            4. 关键标签（4-6个，用逗号分隔）
            """
    )
    reduce_chain = reduce_prompt | llm
    st.session_state["final_report"] = reduce_chain.invoke(
        {"combined_summaries": combined_summaries}
    ).content

# ================= 7. 结果呈现与 RAG 问答区 =================
if "final_report" in st.session_state:
  st.divider()

  # 顶部指标卡
  col1, col2, col3 = st.columns(3)
  col1.metric("解析状态", "成功", delta="100%")
  col2.metric("总字符数", f"{len(st.session_state['raw_text']):,}")
  col3.metric(
      "处理模式",
      "高并发分段提炼"
      if len(st.session_state["raw_text"]) > 3000
      else "单次精炼",
  )

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown("### 📊 结构化分析报告")
  st.markdown(st.session_state["final_report"])

  st.download_button(
      label="📥 导出 Markdown 报告",
      data=st.session_state["final_report"],
      file_name="InsightHub_Report.md",
      mime="text/markdown",
  )

  # RAG 交互问答
  st.divider()
  st.markdown("### 💬 交互式智能问答 (RAG Search)")

  user_query = st.text_input(
      "针对文档细节提问",
      placeholder="例如：这份文档提到的关键指标或核心结论是什么？",
      label_visibility="collapsed",
  )

  if st.button("提交问题", key="qa_btn") and has_api:
    if user_query and "raw_text" in st.session_state:
      with st.spinner("检索中..."):
        qa_llm = ChatOpenAI(model="qwen-plus", temperature=0.3)
        raw_doc = st.session_state["raw_text"]
        qa_prompt = PromptTemplate.from_template(
            """请基于以下提供的文档上下文回答用户的问题。如果文档中未提到，请明确回答“文档中未提及该信息”。

                文档片段上下文：
                {context}

                问题：{question}
                """
        )
        qa_chain = qa_prompt | qa_llm
        answer = qa_chain.invoke(
            {"context": raw_doc[:8000], "question": user_query}
        ).content

        st.info(f"**💡 Insight Engine 回答：**\n\n{answer}")
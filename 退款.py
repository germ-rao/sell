import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st

# ================= 1. 页面基本配置与高级 UI 美化 =================
st.set_page_config(
    page_title="退款申请处理 SOP 决策系统 Demo",
    page_icon="🛡️",
    layout="wide",
)

# 注入自定义 CSS
st.markdown(
    """
    <style>
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


# ================= 3. 核心 SOP 逻辑判断函数 =================
def handle_refund(
    amount: float,
    days: int,
    service_used: bool,
    refund_count: int,
    info_complete: bool,
):
  """本地硬规则规则池（SOP 自动化预审）"""
  is_high_amount = amount > 500

  if not info_complete:
    return (
        "【补充资料】",
        "⚠️ 订单退款申请信息不完整，无法自动校验，已打回要求补充详细凭证。",
    )

  if refund_count >= 2:
    return (
        "【人工审核中】",
        f"🚨 用户近30天退款已达 {refund_count} 次，已触发【风险标记】！无法自动退款，转人工复核是否套刷。",
    )

  if days <= 7 and not service_used:
    if is_high_amount:
      return (
          "【人工审核中】",
          f"⚠️ 满足7天条件，但退款金额（{amount}元）超过 500 元！需提报【管理者/主管】确认审批。",
      )
    else:
      return (
          "【通过】",
          f"✅ 满足7天内未享受核心服务且金额≤500元，已自动触发极速退款并原路退回 {amount} 元！",
      )
  else:
    reason = "超过7天" if days > 7 else "已使用核心服务"
    if is_high_amount:
      return (
          "【人工审核中】",
          f"⚠️ 原因：{reason} 且 金额>500元。已提交人工+管理者联合审核。",
      )
    return (
        "【人工审核中】",
        f"⚠️ 原因：{reason}。已转入一审客服人工审核池，请核实实际履约情况。",
    )


# ================= 4. 头部 Header =================
st.markdown(
    '<div class="main-header">客服新人退款申请处理决策系统 SOP Demo</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">输入退款订单关键字段 • 系统自动模拟 SOP'
    " 逻辑并输出判定状态与操作指引 • AI 辅助话术生成</div>",
    unsafe_allow_html=True,
)

# ================= 5. 表单输入控制区（左右分栏） =================
col1, col2 = st.columns([1, 1])

with col1:
  st.subheader("📝 订单数据输入")

  amount = st.number_input("退款金额 (元)", min_value=0.0, value=180.0, step=10.0)
  days = st.number_input(
      "购买至今天数", min_value=0, value=3, help="7天内支持无理由退款校验"
  )
  refund_count = st.number_input(
      "用户近30天退款次数", min_value=0, value=0
  )

  service_used = st.checkbox("是否已使用核心服务", value=False)
  info_complete = st.checkbox("申请凭证/信息是否完整", value=True)

  submit_btn = st.button("⚡ 运行 SOP 决策引擎", key="process_btn")

# ================= 6. 决策输出与 AI 智能话术 =================
with col2:
  st.subheader("📊 判定结果与 SOP 指引")

  if submit_btn:
    status, detail = handle_refund(
        amount, days, service_used, refund_count, info_complete
    )

    # 显示结果卡片
    if "通过" in status:
      st.success(f"**判定结论：{status}**\n\n{detail}")
    elif "补充" in status:
      st.warning(f"**判定结论：{status}**\n\n{detail}")
    else:
      st.error(f"**判定结论：{status}**\n\n{detail}")

    # 使用大模型（隐藏 API）实时生成客服回复话术
    if has_api:
      st.divider()
      with st.spinner("🤖 AI 正在根据决策结果生成客服推荐回复话术..."):
        llm = ChatOpenAI(model="qwen-plus", temperature=0.5)
        prompt = PromptTemplate.from_template(
            """你是一名经验丰富的高级电商客服团队长。
                系统对该笔退款订单的决策结果如下：
                - 退款金额：{amount} 元
                - 判定结果：{status}
                - 判定依据：{detail}

                请为一线客服生成一段可以直接复制发给用户的、礼貌专业且符合该决策结果的【客服回复话术】。
                要求：语气真诚，说明清楚下一步处理逻辑，100字以内。
                """
        )
        chain = prompt | llm
        ai_reply = chain.invoke(
            {"amount": amount, "status": status, "detail": detail}
        ).content

        st.info(f"**💡 AI 建议客服回复话术：**\n\n{ai_reply}")

      # 保存到 session 防止刷新消失
      st.session_state["last_sop_result"] = {
          "status": status,
          "detail": detail,
          "reply": ai_reply,
      }

  elif "last_sop_result" in st.session_state:
    res = st.session_state["last_sop_result"]
    st.info(f"**上次判定结论：{res['status']}**\n\n{res['detail']}")
    st.caption(f"**AI 建议话术：**\n{res['reply']}")
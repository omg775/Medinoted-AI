with open("app.py", "r") as f:
    text = f.read()

# Replace all standard card instantiations with f-string dynamic classes
text = text.replace(
    "st.markdown('<div class=\"card\">', unsafe_allow_html=True)",
    "st.markdown(f'<div class=\"card {card_class}\">', unsafe_allow_html=True)"
)

# AI Care Assistant explicit target (Line 1287)
text = text.replace(
    "st.markdown('<div class=\"card\" style=\"margin-top: 1rem; padding: 1rem;\">', unsafe_allow_html=True)",
    "st.markdown('<div class=\"card card-ai\" style=\"margin-top: 1rem; padding: 1rem;\">', unsafe_allow_html=True)"
)

# Inject emojis to section headers
text = text.replace('<div class="section-header">Behavioral Health Synthesis</div>', '<div class="section-header">💡 Behavioral Health Synthesis</div>')
text = text.replace('<div class="section-header">Semantic Entity Extraction</div>', '<div class="section-header">🧬 Semantic Entity Extraction</div>')
text = text.replace('<div class="section-header">SOAP Note History</div>', '<div class="section-header">📄 SOAP Note History</div>')
text = text.replace('<div class="section-header">Historical Data Streams</div>', '<div class="section-header">📈 Historical Data Streams</div>')
text = text.replace('<div class="section-header">Predictive Risk Intelligence</div>', '<div class="section-header">⚠️ Predictive Risk Intelligence</div>')
text = text.replace('<div class="section-header">Behavioral Sentiment Mapping</div>', '<div class="section-header">🗺️ Behavioral Sentiment Mapping</div>')
text = text.replace('<div class="section-header">Symptomatic Trend Analysis</div>', '<div class="section-header">🔍 Symptomatic Trend Analysis</div>')
text = text.replace('<div class="section-header">SOAP Protocol Output</div>', '<div class="section-header">📋 SOAP Protocol Output</div>')
text = text.replace('<div class="section-header">Clinical Protocol Validation</div>', '<div class="section-header">✅ Clinical Protocol Validation</div>')


# Improve chat bubbles representation. Currently the AI assistant bubbles just use normal text. Wait, we don't have explicit chat bubble divs in the python code.
# The code is:
# with st.chat_message(msg["role"]):
#     st.write(msg["content"])
# st.chat_message natively adds generic streamlit styling.
# I will overwrite it slightly to inject my CSS directly if I can, OR since streamlit already outputs classes for chat messages I don't need to do anything since the CSS targets '.chat-bubble-...'.
# Actually, the CSS generated was `.chat-bubble-assistant` and `.chat-bubble-user` but I don't see those classes assigned in the UI. 
# Streamlit native chat messages use data-testid. 
# I will use JS injected to style them or alter the CSS selector.
# Or better, output explicit html blocks instead of `st.chat_message` if the user wants custom chat bubbles. Or use streamlit's `st.markdown('<div class="chat-bubble-user">...</div>', unsafe_allow_html=True)`.

with open("app.py", "w") as f:
    f.write(text)

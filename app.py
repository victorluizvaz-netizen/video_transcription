import streamlit as st
import whisper
import tempfile
import os
from openai import OpenAI

# 1. Teste de Importação
try:
    from openai import OpenAI
    openai_instalação = True
except ImportError:
    openai_instalação = False

st.set_page_config(page_title="Editor de Transcrição", layout="wide")

# 2. Teste de Chave
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("❌ Chave OPENAI_API_KEY não encontrada nas Secrets!")
    st.stop()

if not openai_instalação:
    st.error("❌ A biblioteca OpenAI não foi instalada. Mova o 'requirements.txt' para a raiz do GitHub.")
    st.stop()

st.title("🎙️ Transcritor Profissional")

# --- Interface Simples ---
arquivo_video = st.sidebar.file_uploader("Upload", type=["mp4", "mov", "mkv"])

if arquivo_video:
    st.video(arquivo_video)
    if st.button("🚀 Iniciar Processo"):
        with st.spinner("Transcrevendo..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(arquivo_video.read())
                path = tmp.name
            
            try:
                model = whisper.load_model("base")
                result = model.transcribe(path)
                st.session_state['txt'] = result["text"]
                st.success("Transcrição concluída!")
            finally:
                if os.path.exists(path): os.remove(path)

    if 'txt' in st.session_state:
        st.subheader("📝 Texto e Comandos")
        cmd = st.text_input("O que fazer? (ex: Corrija o texto)")
        if st.button("Enviar para ChatGPT"):
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"{cmd}\n\nTexto: {st.session_state['txt']}"}]
            )
            st.write(res.choices[0].message.content)

import streamlit as st
import whisper
import tempfile
import os
import google.generativeai as genai

# Configuração da página
st.set_page_config(page_title="Transcritor & Resumidor IA", layout="wide", page_icon="🎙️")

# Configuração da API do Gemini (Pegando das Secrets do Streamlit)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Por favor, configure a variável GEMINI_API_KEY nas Secrets do Streamlit.")

def gerar_resumo(texto):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Com base na seguinte transcrição de vídeo, crie um resumo executivo com os pontos principais em bullet points e uma conclusão curta: {texto}"
    response = model.generate_content(prompt)
    return response.text

st.title("🎙️ Transcritor Inteligente com Resumo IA")
st.markdown("---")

# Barra Lateral
st.sidebar.header("Configurações")
arquivo_video = st.sidebar.file_uploader("Upload do Vídeo", type=["mp4", "mov", "mkv"])
modelo_ia = st.sidebar.selectbox("Modelo Whisper (Precisão)", ["base", "small", "medium"])

if arquivo_video:
    st.video(arquivo_video)
    
    if st.button("🚀 Processar Vídeo"):
        with st.spinner("1. Transcrevendo áudio (isso pode demorar dependendo do tamanho)..."):
            # Salva temp
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(arquivo_video.read())
                temp_path = tfile.name

            # Transcrição
            model = whisper.load_model(modelo_ia)
            result = model.transcribe(temp_path)
            texto_transcrito = result["text"]
            
        st.success("✅ Transcrição Concluída!")
        
        # Abas para organizar o conteúdo
        tab1, tab2 = st.tabs(["📝 Transcrição Completa", "💡 Resumo IA"])
        
        with tab1:
            st.text_area("Texto na íntegra:", value=texto_transcrito, height=400)
            st.download_button("Baixar Transcrição", texto_transcrito, file_name="transcricao.txt")
            
        with tab2:
            with st.spinner("Gerando resumo com Gemini..."):
                try:
                    resumo = gerar_resumo(texto_transcrito)
                    st.markdown(resumo)
                    st.download_button("Baixar Resumo", resumo, file_name="resumo.txt")
                except Exception as e:
                    st.error(f"Erro ao gerar resumo: {e}")
        
        os.remove(temp_path)
else:
    st.info("💡 Faça o upload de um vídeo para começar.")
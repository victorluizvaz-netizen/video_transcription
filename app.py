import streamlit as st
import whisper
import tempfile
import os
from groq import Groq

# --- CONFIGURAÇÕES INICIAIS ---
st.set_page_config(page_title="Transcritor IA Profissional", layout="wide", page_icon="🎙️")

# Configuração da API Groq (Substituindo OpenAI/Gemini)
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("Configure a variável GROQ_API_KEY nas Secrets do Streamlit.")
    st.stop()

# --- FUNÇÃO DE PROCESSAMENTO COM GROQ (LLAMA 3) ---
def processar_com_ia(texto, instrucao):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Modelo potente e gratuito no Groq
            messages=[
                {"role": "system", "content": "Você é um editor especializado em transcrições profissionais."},
                {"role": "user", "content": f"Instrução: {instrucao}\n\nTexto:\n{texto}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na IA: {str(e)}"

# --- INTERFACE ---
st.title("🎙️ Transcritor & Editor IA (Grátis via Groq)")

st.sidebar.header("1. Configurações")
arquivo_video = st.sidebar.file_uploader("Upload do Vídeo", type=["mp4", "mov", "mkv"])
modelo_ia = st.sidebar.selectbox("Precisão do Whisper", ["base", "small"], index=1)

if 'transcricao' not in st.session_state: st.session_state['transcricao'] = None
if 'resultado_ia' not in st.session_state: st.session_state['resultado_ia'] = None

# --- FLUXO PRINCIPAL ---
if arquivo_video:
    st.video(arquivo_video)
    
    if st.button("🚀 Iniciar Transcrição"):
        st.session_state['transcricao'] = None
        st.session_state['resultado_ia'] = None
        
        with st.spinner("Whisper transcrevendo..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(arquivo_video.read())
                temp_path = tmp.name

            try:
                model = whisper.load_model(modelo_ia)
                result = model.transcribe(temp_path)
                st.session_state['transcricao'] = result["text"]
                st.success("✅ Transcrição concluída!")
            except Exception as e:
                st.error(f"Erro no Whisper: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)

    if st.session_state['transcricao']:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Transcrição Original")
            st.text_area("Texto Bruto:", value=st.session_state['transcricao'], height=450)
            st.download_button("Baixar Original", st.session_state['transcricao'], file_name="original.txt")

        with col2:
            st.subheader("🪄 Editor Inteligente")
            instrucao_usuario = st.text_input("O que deseja fazer?", placeholder="Ex: Melhore o texto e organize...")
            
            c1, c2, c3 = st.columns(3)
            if c1.button("✨ Refinar"):
                with st.spinner("IA processando..."):
                    st.session_state['resultado_ia'] = processar_com_ia(st.session_state['transcricao'], "Corrija concordância e organize em parágrafos.")
            
            if c2.button("💡 Resumir"):
                with st.spinner("IA processando..."):
                    st.session_state['resultado_ia'] = processar_com_ia(st.session_state['transcricao'], "Crie um resumo em bullet points.")
            
            if c3.button("🌍 Traduzir EN"):
                with st.spinner("IA processando..."):
                    st.session_state['resultado_ia'] = processar_com_ia(st.session_state['transcricao'], "Traduza para o inglês.")

            if st.button("Enviar Comando"):
                if instrucao_usuario:
                    with st.spinner("IA processando..."):
                        st.session_state['resultado_ia'] = processar_com_ia(st.session_state['transcricao'], instrucao_usuario)

            if st.session_state['resultado_ia']:
                st.markdown("---")
                st.write(st.session_state['resultado_ia'])
                st.download_button("Baixar Editado", st.session_state['resultado_ia'], file_name="resultado.txt")
else:
    st.info("Aguardando upload de vídeo.")

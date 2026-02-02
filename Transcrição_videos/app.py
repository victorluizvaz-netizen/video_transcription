import streamlit as st
import whisper
import tempfile
import os
from openai import OpenAI

# --- CONFIGURAÇÕES INICIAIS ---
st.set_page_config(page_title="Transcritor IA Profissional", layout="wide", page_icon="🎙️")

# Configuração da API OpenAI
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("Configure a variável OPENAI_API_KEY nas Secrets do Streamlit.")

# --- FUNÇÃO DE PROCESSAMENTO COM CHATGPT ---
def processar_com_gpt(texto, instrucao):
    response = client.chat.completions.create(
        model="gpt-4o-mini", # Modelo rápido e barato
        messages=[
            {"role": "system", "content": "Você é um assistente editorial profissional especializado em transcrições."},
            {"role": "user", "content": f"Instrução: {instrucao}\n\nTexto:\n{texto}"}
        ]
    )
    return response.choices[0].message.content

# --- INTERFACE ---
st.title("🎙️ Transcritor & Editor IA")
st.markdown("Transcreva vídeos e use o ChatGPT para refinar o conteúdo como desejar.")

# Barra Lateral
st.sidebar.header("1. Configurações")
arquivo_video = st.sidebar.file_uploader("Upload do Vídeo", type=["mp4", "mov", "mkv"])
modelo_ia = st.sidebar.selectbox("Precisão do Whisper", ["base", "small"], index=1)

# Estado da Sessão
if 'transcricao' not in st.session_state:
    st.session_state['transcricao'] = None
if 'resultado_gpt' not in st.session_state:
    st.session_state['resultado_gpt'] = None

# --- FLUXO PRINCIPAL ---
if arquivo_video:
    st.video(arquivo_video)
    
    if st.button("🚀 Iniciar Transcrição"):
        with st.spinner("Extraindo áudio e transcrevendo..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(arquivo_video.read())
                tmp_path = tmp.name

            try:
                model = whisper.load_model(modelo_ia)
                result = model.transcribe(tmp_path)
                st.session_state['transcricao'] = result["text"]
                st.success("✅ Transcrição concluída!")
            except Exception as e:
                st.error(f"Erro no Whisper: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if st.session_state['transcricao']:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Transcrição Original")
            st.text_area("Bruto:", value=st.session_state['transcricao'], height=400)
            st.download_button("Baixar Bruto", st.session_state['transcricao'], file_name="bruto.txt")

        with col2:
            st.subheader("🪄 Refinamento e Comandos")
            # Campo de solicitações personalizadas
            instrucao_usuario = st.text_input(
                "O que deseja fazer com o texto?", 
                placeholder="Ex: Corrija a gramática e organize em parágrafos..."
            )
            
            if st.button("Executar Comando IA"):
                if instrucao_usuario:
                    with st.spinner("O ChatGPT está processando..."):
                        try:
                            st.session_state['resultado_gpt'] = processar_com_gpt(
                                st.session_state['transcricao'], 
                                instrucao_usuario
                            )
                        except Exception as e:
                            st.error(f"Erro na OpenAI: {e}")
                else:
                    st.warning("Por favor, digite uma instrução.")

            if st.session_state['resultado_gpt']:
                st.text_area("Resultado IA:", value=st.session_state['resultado_gpt'], height=300)
                st.download_button("Baixar Resultado", st.session_state['resultado_gpt'], file_name="ia_resultado.txt")
else:
    st.info("Faça o upload de um vídeo para começar.")

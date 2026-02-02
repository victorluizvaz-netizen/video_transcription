import streamlit as st
import whisper
import tempfile
import os
import google.generativeai as genai

# --- CONFIGURAÇÕES INICIAIS ---
st.set_page_config(page_title="Transcritor IA Profissional", layout="wide", page_icon="🎙️")

# Configuração da API do Gemini
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Configure a variável GEMINI_API_KEY nas Secrets do Streamlit.")

# --- FUNÇÕES DE APOIO ---
def gerar_resumo(texto):
    # Atualizado para gemini-1.5-flash
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Com base na seguinte transcrição, crie um resumo executivo com pontos principais (bullet points) e uma conclusão curta:\n\n{texto}"
    response = model.generate_content(prompt)
    return response.text

def refinar_texto(texto):
    # Atualizado para gemini-1.5-flash
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = (
        "Você é um editor profissional. Re-escreva a transcrição a seguir para que fique clara, fluida e profissional. "
        "Corrija erros de concordância, melhore a pontuação e organize o texto em parágrafos com indentação correta. "
        "Mantenha o sentido original. Texto:\n\n" + texto
    )
    response = model.generate_content(prompt)
    return response.text

# --- INTERFACE ---
st.title("🎙️ Transcritor & Assistente de Conteúdo")
st.markdown("Converta vídeos em texto e utilize IA para refinar ou resumir o conteúdo.")

# Barra Lateral
st.sidebar.header("1. Upload")
arquivo_video = st.sidebar.file_uploader("Escolha um vídeo", type=["mp4", "mov", "mkv"])

# Nota sobre memória: Mantendo base e small para evitar crash no Streamlit Cloud
modelo_ia = st.sidebar.selectbox("Precisão do Whisper", ["base", "small"], index=1)

# Inicialização do Estado da Sessão
if 'transcricao' not in st.session_state:
    st.session_state['transcricao'] = None
if 'refinado' not in st.session_state:
    st.session_state['refinado'] = None
if 'resumo' not in st.session_state:
    st.session_state['resumo'] = None

# --- FLUXO PRINCIPAL ---
if arquivo_video:
    st.video(arquivo_video)
    
    if st.button("🚀 Iniciar Transcrição"):
        # Limpa estados anteriores para novo vídeo
        st.session_state['refinado'] = None
        st.session_state['resumo'] = None
        
        with st.spinner("Processando vídeo... (Aguarde a conclusão)"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(arquivo_video.read())
                tmp_path = tmp.name

            try:
                model = whisper.load_model(modelo_ia)
                result = model.transcribe(tmp_path)
                st.session_state['transcricao'] = result["text"]
                st.success("✅ Transcrição concluída!")
            except Exception as e:
                st.error(f"Erro no processamento: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if st.session_state['transcricao']:
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📝 Transcrição Bruta", "🪄 Melhorar Texto (IA)", "💡 Resumo Executivo"])

        with tab1:
            st.text_area("Texto original:", value=st.session_state['transcricao'], height=300)
            st.download_button("Baixar Bruto (.txt)", st.session_state['transcricao'], file_name="bruto.txt")

        with tab2:
            st.markdown("### Refinamento de Texto")
            if st.button("✨ Aplicar Melhorias"):
                with st.spinner("O Gemini está editando seu texto..."):
                    try:
                        st.session_state['refinado'] = refinar_texto(st.session_state['transcricao'])
                    except Exception as e:
                        st.error(f"Erro ao refinar: {e}")
            
            if st.session_state['refinado']:
                st.text_area("Texto Refinado:", value=st.session_state['refinado'], height=300)
                st.download_button("Baixar Refinado (.txt)", st.session_state['refinado'], file_name="refinado.txt")

        with tab3:
            st.markdown("### Resumo Gerado")
            if st.button("📝 Gerar Resumo"):
                with st.spinner("Analisando conteúdo..."):
                    try:
                        st.session_state['resumo'] = gerar_resumo(st.session_state['transcricao'])
                    except Exception as e:
                        st.error(f"Erro ao resumir: {e}")
            
            if st.session_state['resumo']:
                st.markdown(st.session_state['resumo'])
                st.download_button("Baixar Resumo (.txt)", st.session_state['resumo'], file_name="resumo.txt")
else:
    st.info("Arraste um vídeo para a barra lateral para começar.")

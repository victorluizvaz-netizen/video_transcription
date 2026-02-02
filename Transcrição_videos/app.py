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

# --- FUNÇÕES DE APOIO (MODELO ATUALIZADO) ---
def gerar_resumo(texto):
    # Usando o modelo mais atual e estável para evitar erro 404
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Com base na seguinte transcrição, crie um resumo executivo com pontos principais (bullet points) e uma conclusão curta:\n\n{texto}"
    response = model.generate_content(prompt)
    return response.text

def refinar_texto(texto):
    # Usando o modelo mais atual e estável para evitar erro 404
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
st.sidebar.header("1. Configurações")
arquivo_video = st.sidebar.file_uploader("Upload do Vídeo", type=["mp4", "mov", "mkv"])

# Nota: Mantendo 'base' e 'small' para evitar crash de RAM no Streamlit Cloud
modelo_ia = st.sidebar.selectbox(
    "Precisão do Whisper", 
    ["base", "small"], 
    index=1,
    help="O modelo 'medium' é pesado demais para o servidor gratuito do Streamlit."
)

# Inicialização do Estado da Sessão (Preserva os dados entre cliques)
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
        # Limpa estados para um novo processamento
        st.session_state['transcricao'] = None
        st.session_state['refinado'] = None
        st.session_state['resumo'] = None
        
        with st.spinner("Extraindo áudio e transcrevendo..."):
            # Criação de arquivo temporário robusto
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(arquivo_video.read())
                tmp_path = tmp.name

            try:
                # Carregamento do modelo Whisper
                model = whisper.load_model(modelo_ia)
                result = model.transcribe(tmp_path)
                st.session_state['transcricao'] = result["text"]
                st.success("✅ Transcrição concluída!")
            except Exception as e:
                st.error(f"Erro no processamento do vídeo: {e}")
            finally:
                # Limpeza obrigatória do arquivo temporário
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # Exibição de abas caso já exista transcrição
    if st.session_state['transcricao']:
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📝 Transcrição Bruta", "🪄 Refinar Texto", "💡 Resumo IA"])

        with tab1:
            st.text_area("Original:", value=st.session_state['transcricao'], height=350)
            st.download_button("Baixar Bruto (.txt)", st.session_state['transcricao'], file_name="transcricao_bruta.txt")

        with tab2:
            st.markdown("### ✍️ Refinamento Profissional")
            if st.button("✨ Melhorar Texto"):
                with st.spinner("O Gemini está editando seu texto..."):
                    try:
                        st.session_state['refinado'] = refinar_texto(st.session_state['transcricao'])
                    except Exception as e:
                        st.error(f"Erro na API do Gemini: {e}")
            
            if st.session_state['refinado']:
                st.text_area("Texto Refinado:", value=st.session_state['refinado'], height=350)
                st.download_button("Baixar Refinado (.txt)", st.session_state['refinado'], file_name="texto_refinado.txt")

        with tab3:
            st.markdown("### 💡 Resumo dos Pontos Chave")
            if st.button("📝 Gerar Resumo"):
                with st.spinner("Analisando transcrição..."):
                    try:
                        st.session_state['resumo'] = gerar_resumo(st.session_state['transcricao'])
                    except Exception as e:
                        st.error(f"Erro na API do Gemini: {e}")
            
            if st.session_state['resumo']:
                st.markdown(st.session_state['resumo'])
                st.download_button("Baixar Resumo (.txt)", st.session_state['resumo'], file_name="resumo_executivo.txt")
else:
    st.info("Faça o upload de um vídeo para começar.")

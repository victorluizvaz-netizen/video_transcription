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
    st.error("ERRO: Configure a variável OPENAI_API_KEY nas Secrets do Streamlit.")
    st.stop() # Interrompe a execução se não houver chave

# --- FUNÇÃO DE PROCESSAMENTO COM CHATGPT ---
def processar_com_gpt(texto, instrucao):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente editorial profissional. Melhore o texto conforme solicitado pelo usuário, mantendo a clareza e o profissionalismo."},
                {"role": "user", "content": f"Texto Original:\n{texto}\n\nSolicitação específica do usuário: {instrucao}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na OpenAI: {str(e)}"

# --- INTERFACE ---
st.title("🎙️ Transcritor & Editor IA")
st.markdown("Converta vídeos em texto e use IA para refinar o conteúdo.")

# Barra Lateral
st.sidebar.header("1. Configurações")
arquivo_video = st.sidebar.file_uploader("Upload do Vídeo", type=["mp4", "mov", "mkv"])
modelo_ia = st.sidebar.selectbox("Precisão do Whisper", ["base", "small"], index=1)

# Estado da Sessão (Preserva dados entre cliques)
if 'transcricao' not in st.session_state:
    st.session_state['transcricao'] = None
if 'resultado_gpt' not in st.session_state:
    st.session_state['resultado_gpt'] = None

# --- FLUXO PRINCIPAL ---
if arquivo_video:
    st.video(arquivo_video)
    
    if st.button("🚀 Iniciar Transcrição"):
        # Limpa estados para novo vídeo
        st.session_state['transcricao'] = None
        st.session_state['resultado_gpt'] = None
        
        with st.spinner("Extraindo áudio e transcrevendo..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(arquivo_video.read())
                temp_path = tmp.name

            try:
                # O Whisper precisa do ffmpeg (que está no packages.txt)
                model = whisper.load_model(modelo_ia)
                result = model.transcribe(temp_path)
                st.session_state['transcricao'] = result["text"]
                st.success("✅ Transcrição concluída!")
            except Exception as e:
                st.error(f"Erro no Whisper: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # Exibição dos resultados
    if st.session_state['transcricao']:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Transcrição Original")
            st.text_area("Bruto:", value=st.session_state['transcricao'], height=400)
            st.download_button("Baixar Bruto (.txt)", st.session_state['transcricao'], file_name="original.txt")

        with col2:
            st.subheader("🪄 Editor Inteligente")
            # Campo de solicitações customizadas
            instrucao_usuario = st.text_input(
                "O que deseja fazer?", 
                placeholder="Ex: Corrija concordância, crie tópicos, resuma..."
            )
            
            # Botões rápidos (Opcional - facilitam o uso)
            c1, c2, c3 = st.columns(3)
            if c1.button("✨ Refinar"): instrucao_usuario = "Corrija concordância, pontuação e organize em parágrafos."
            if c2.button("💡 Resumir"): instrucao_usuario = "Crie um resumo em bullet points com os pontos principais."
            if c3.button("🌍 Traduzir EN"): instrucao_usuario = "Traduza o texto fielmente para o Inglês."

            if st.button("Executar com ChatGPT"):
                if instrucao_usuario:
                    with st.spinner("IA processando..."):
                        st.session_state['resultado_gpt'] = processar_com_gpt(
                            st.session_state['transcricao'], 
                            instrucao_usuario
                        )
                else:
                    st.warning("Escreva uma instrução ou use um botão rápido.")

            if st.session_state['resultado_gpt']:
                st.text_area("Resultado IA:", value=st.session_state['resultado_gpt'], height=300)
                st.download_button("Baixar Editado (.txt)", st.session_state['resultado_gpt'], file_name="editado.txt")
else:
    st.info("Aguardando upload de vídeo.")

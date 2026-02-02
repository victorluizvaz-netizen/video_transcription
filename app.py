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
    st.stop()

# --- FUNÇÃO DE PROCESSAMENTO COM CHATGPT ---
def processar_com_gpt(texto, instrucao):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente editorial profissional especializado em transcrições."},
                {"role": "user", "content": f"Instrução: {instrucao}\n\nTexto:\n{texto}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na OpenAI: {str(e)}"

# --- INTERFACE ---
st.title("🎙️ Transcritor & Editor IA Profissional")

# Barra Lateral
st.sidebar.header("1. Configurações")
arquivo_video = st.sidebar.file_uploader("Upload do Vídeo", type=["mp4", "mov", "mkv"])

# Restaurando o seletor de precisão
modelo_ia = st.sidebar.selectbox(
    "Precisão do Whisper", 
    ["base", "small"], 
    index=1,
    help="O modelo 'small' é mais lento porém mais preciso. O 'base' é mais rápido."
)

# Inicialização do Estado da Sessão (Fundamental para o texto não sumir)
if 'transcricao' not in st.session_state:
    st.session_state['transcricao'] = None
if 'resultado_gpt' not in st.session_state:
    st.session_state['resultado_gpt'] = None

# --- FLUXO PRINCIPAL ---
if arquivo_video:
    st.video(arquivo_video)
    
    if st.button("🚀 Iniciar Transcrição"):
        # Limpa estados para um novo vídeo
        st.session_state['transcricao'] = None
        st.session_state['resultado_gpt'] = None
        
        with st.spinner(f"O Whisper está processando com o modelo '{modelo_ia}'..."):
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
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # Exibição dos resultados (SÓ APARECE SE TIVER TEXTO TRANSCRITO)
    if st.session_state['transcricao']:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Transcrição Original")
            # Aqui você vê o texto bruto gerado pelo Whisper
            st.text_area("Texto Bruto:", value=st.session_state['transcricao'], height=450)
            st.download_button("Baixar Original (.txt)", st.session_state['transcricao'], file_name="original.txt")

        with col2:
            st.subheader("🪄 Editor ChatGPT")
            instrucao_usuario = st.text_input(
                "O que deseja fazer com este texto?", 
                placeholder="Ex: Corrija concordância, resuma em tópicos, traduza..."
            )
            
            # Botões rápidos para facilitar sua vida
            st.markdown("**Sugestões rápidas:**")
            c1, c2, c3 = st.columns(3)
            
            # Lógica dos botões de atalho
            if c1.button("✨ Refinar Texto"):
                with st.spinner("Refinando..."):
                    st.session_state['resultado_gpt'] = processar_com_gpt(st.session_state['transcricao'], "Corrija concordância, pontuação e organize em parágrafos profissionais.")
            
            if c2.button("💡 Resumir"):
                with st.spinner("Resumindo..."):
                    st.session_state['resultado_gpt'] = processar_com_gpt(st.session_state['transcricao'], "Crie um resumo executivo em bullet points.")
            
            if c3.button("🌍 Traduzir EN"):
                with st.spinner("Traduzindo..."):
                    st.session_state['resultado_gpt'] = processar_com_gpt(st.session_state['transcricao'], "Traduza este texto para o inglês profissional.")

            # Botão para o comando customizado do campo de texto
            if st.button("Executar Comando Personalizado"):
                if instrucao_usuario:
                    with st.spinner("Processando solicitação..."):
                        st.session_state['resultado_gpt'] = processar_com_gpt(st.session_state['transcricao'], instrucao_usuario)
                else:
                    st.warning("Digite algo no campo acima.")

            # Exibe o resultado da IA se houver
            if st.session_state['resultado_gpt']:
                st.markdown("---")
                st.markdown("### 🤖 Resultado da IA:")
                st.write(st.session_state['resultado_gpt'])
                st.download_button("Baixar Resultado (.txt)", st.session_state['resultado_gpt'], file_name="resultado_ia.txt")
else:
    st.info("Aguardando upload do vídeo na barra lateral.")

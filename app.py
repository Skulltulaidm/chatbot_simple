import streamlit as st

st.set_page_config(
    page_title="Chatbot con Análisis de Archivos",
    page_icon="🤖",
    layout="wide"
)

import pandas as pd
from typing import Optional, List, Dict, Any
import json
import base64
from io import BytesIO
import tempfile
import os
from datetime import datetime

from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF para PDFs
import docx


class ChatbotApp:
    def __init__(self):
        self.initialize_session_state()
        self.llm = self.get_llm()

    def initialize_session_state(self):
        """Inicializa el estado de la sesión"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        if "file_contents" not in st.session_state:
            st.session_state.file_contents = {}

    @st.cache_resource
    def get_llm(_self) -> Optional[AzureChatOpenAI]:
        """
        Inicializa y retorna la instancia de Azure OpenAI.
        """
        try:
            return AzureChatOpenAI(
                azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
                deployment_name=st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"],
                api_key=st.secrets["AZURE_OPENAI_API_KEY"],
                api_version="2023-07-01-preview",
                temperature=0.1,
                streaming=True
            )
        except Exception as e:
            st.error(f"Error al configurar Azure OpenAI: {str(e)}")
            return None

    def process_pdf(self, file) -> str:
        """Procesa archivos PDF y extrae el texto"""
        try:
            # Guardar temporalmente el archivo
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name

            # Extraer texto usando PyMuPDF
            doc = fitz.open(tmp_file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Limpiar archivo temporal
            os.unlink(tmp_file_path)

            return text
        except Exception as e:
            st.error(f"Error procesando PDF: {str(e)}")
            return ""

    def process_docx(self, file) -> str:
        """Procesa archivos DOCX y extrae el texto"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error procesando DOCX: {str(e)}")
            return ""

    def process_excel(self, file) -> str:
        """Procesa archivos Excel y convierte a texto"""
        try:
            # Leer todas las hojas del Excel
            excel_data = pd.read_excel(file, sheet_name=None)
            text = ""

            for sheet_name, df in excel_data.items():
                text += f"\n--- Hoja: {sheet_name} ---\n"
                text += df.to_string(index=False)
                text += "\n\n"

            return text
        except Exception as e:
            st.error(f"Error procesando Excel: {str(e)}")
            return ""

    def process_image(self, file) -> str:
        """Procesa imágenes y las convierte a base64 para análisis"""
        try:
            image = Image.open(file)

            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Redimensionar si es muy grande
            max_size = (1024, 1024)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Convertir a base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return f"Imagen procesada: {file.name} (dimensiones: {image.size})"
        except Exception as e:
            st.error(f"Error procesando imagen: {str(e)}")
            return ""

    def process_uploaded_files(self, uploaded_files) -> Dict[str, str]:
        """Procesa todos los archivos subidos"""
        file_contents = {}

        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            file_name = uploaded_file.name

            with st.spinner(f"Procesando {file_name}..."):
                if file_type == "application/pdf":
                    content = self.process_pdf(uploaded_file)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    content = self.process_docx(uploaded_file)
                elif file_type in ["application/vnd.ms-excel",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                    content = self.process_excel(uploaded_file)
                elif file_type.startswith("image/"):
                    content = self.process_image(uploaded_file)
                else:
                    content = f"Tipo de archivo no soportado: {file_type}"

                file_contents[file_name] = content

        return file_contents

    def get_response(self, user_input: str, file_context: str = "") -> str:
        """Obtiene respuesta del modelo con contexto de archivos"""
        if not self.llm:
            return "Error: No se pudo conectar con Azure OpenAI"

        try:
            # Construir prompt con contexto de archivos si existe
            prompt = user_input
            if file_context:
                prompt = f"""
Contexto de archivos analizados:
{file_context}

Pregunta del usuario: {user_input}

Por favor, responde considerando el contexto de los archivos proporcionados.
"""

            # Obtener historial de conversación
            chat_history = st.session_state.memory.chat_memory.messages

            # Crear lista de mensajes para el modelo
            messages = []
            for msg in chat_history:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})

            # Agregar mensaje actual
            messages.append({"role": "user", "content": prompt})

            # Obtener respuesta
            response = self.llm.invoke(messages)

            # Guardar en memoria
            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(response.content)

            return response.content

        except Exception as e:
            st.error(f"Error obteniendo respuesta: {str(e)}")
            return "Lo siento, hubo un error al procesar tu solicitud."

    def clear_memory(self):
        """Limpia la memoria de conversación"""
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.session_state.file_contents = {}
        st.success("Conversación y archivos limpiados")

    def export_conversation(self) -> str:
        """Exporta la conversación actual"""
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "files_processed": list(st.session_state.file_contents.keys())
        }
        return json.dumps(conversation_data, indent=2, ensure_ascii=False)

    def run(self):
        """Ejecuta la aplicación principal"""

        st.title("🤖 Chatbot con Análisis de Archivos")
        st.markdown("---")

        # Sidebar para configuración y archivos
        with st.sidebar:
            st.header("📁 Subir Archivos")

            uploaded_files = st.file_uploader(
                "Selecciona archivos para analizar:",
                type=['pdf', 'docx', 'xlsx', 'xls', 'png', 'jpg', 'jpeg', 'gif'],
                accept_multiple_files=True,
                help="Soporta: PDF, DOCX, Excel, e imágenes"
            )

            if uploaded_files:
                new_file_contents = self.process_uploaded_files(uploaded_files)
                st.session_state.file_contents.update(new_file_contents)

                st.success(f"✅ {len(uploaded_files)} archivo(s) procesado(s)")

                # Mostrar archivos procesados
                if st.session_state.file_contents:
                    st.subheader("📋 Archivos Procesados:")
                    for filename in st.session_state.file_contents.keys():
                        st.write(f"• {filename}")

            st.markdown("---")

            # Controles de conversación
            st.header("⚙️ Controles")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Limpiar Chat", use_container_width=True):
                    self.clear_memory()
                    st.rerun()

            with col2:
                if st.button("📥 Exportar", use_container_width=True):
                    if st.session_state.messages:
                        conversation_json = self.export_conversation()
                        st.download_button(
                            label="Descargar Conversación",
                            data=conversation_json,
                            file_name=f"conversacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

        # Área principal de chat
        st.header("💬 Conversación")

        # Contenedor para mensajes
        chat_container = st.container()

        with chat_container:
            # Mostrar historial de mensajes
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Input de usuario
        if prompt := st.chat_input("Escribe tu mensaje aquí..."):
            # Agregar mensaje del usuario
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generar respuesta
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    # Preparar contexto de archivos
                    file_context = ""
                    if st.session_state.file_contents:
                        file_context = "\n\n".join([
                            f"Archivo: {filename}\nContenido:\n{content}"
                            for filename, content in st.session_state.file_contents.items()
                        ])

                    response = self.get_response(prompt, file_context)
                    st.markdown(response)

            # Agregar respuesta al historial
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Información sobre el estado
        if st.session_state.file_contents:
            st.info(f"📊 Archivos en contexto: {len(st.session_state.file_contents)}")


if __name__ == "__main__":
    app = ChatbotApp()
    app.run()

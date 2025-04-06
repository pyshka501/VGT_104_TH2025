import os
import glob
import logging
import pandas as pd
from typing import List, Optional, Dict, Union, Any, Tuple
from tqdm import tqdm
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
import markdown
import yaml
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

class ConfigLoader:
    DEFAULT_CONFIG = {
        'embedding': {
            'model_name': "thenlper/gte-small",
            'chunk_size': 512,
            'chunk_overlap': 100,
            'device': "cuda"
        },
        'llm': {
            'model_name': "Qwen/Qwen2.5-14B-Instruct",
            'max_new_tokens': 512,
            'quantization': None,
            'system_prompt': "You are an experienced technical support specialist..."
        },
        'processing': {
            'pdf_chunk_size': 1000,
            'pdf_chunk_overlap': 200,
            'persist_directory': "./chroma_db",
            'num_workers': 4
        },
        'data_sources': {
            'pdf_directory': "data/pdfs",
            'excel_path': "data/test_article.xlsx"
        },
        'excel_processing': {
            'query_column': 'question',
            'answer_column': 'response',
            'sheet_names': None,
            'max_rows': 1000,
            'text_columns': ['answer', 'additional_info'],
            'metadata_columns': ['category', 'priority'],
            'include_query_in_answer': True,
            'chunk_size': 768,
            'chunk_overlap': 128
        }
    }
    
    def __init__(self, config_path: str = "config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path) as f:
                user_config = yaml.safe_load(f)
                return self._validate_config(self._deep_merge(self.DEFAULT_CONFIG.copy(), user_config))
        return self._validate_config(self.DEFAULT_CONFIG.copy())

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                base[key] = self._deep_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    def _validate_config(self, config: Dict) -> Dict:
        required_fields = {
            'embedding': ['model_name', 'chunk_size'],
            'llm': ['model_name'],
            'processing': ['persist_directory']
        }
        
        for section, fields in required_fields.items():
            if section not in config:
                raise ValueError(f"Missing section {section} in config")
            for field in fields:
                if field not in config[section]:
                    raise ValueError(f"Missing field {field} in section {section}")
        return config

class DocumentProcessor:
    def __init__(self, pdf_chunk_size: int = 1000, pdf_chunk_overlap: int = 200,
                 excel_config: Optional[Dict] = None):
        self.pdf_chunk_size = pdf_chunk_size
        self.pdf_chunk_overlap = pdf_chunk_overlap
        self.excel_config = excel_config or {}

    
    def _clean_metadata(self, metadata: Dict) -> Dict:
        return {k: str(v) if isinstance(v, list) else v for k, v in metadata.items()}

    
    
    def process_pdf(self, file_path: str) -> List[LangchainDocument]:
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            filename = Path(file_path).name
            return [LangchainDocument(
                page_content=page.page_content,
                metadata={
                    "source": filename,
                    "page": page.metadata.get('page', '') + 1,  # Нумерация с 1
                    "doc_type": "PDF"
                }
            ) for page in pages]
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            return []

    def process_excel(self, excel_path: str) -> List[LangchainDocument]:
        try:
            docs = []
            sheet_names = self._get_sheet_names(excel_path)
            excel_filename = Path(excel_path).name
            
            for sheet_name in sheet_names:
                df = pd.read_excel(
                    excel_path,
                    sheet_name=sheet_name,
                    nrows=self.excel_config.get('max_rows')
                )
                df = self._preprocess_dataframe(df, sheet_name)
                
                source_name = f"{excel_filename} (лист: {sheet_name})"
                docs.extend(self._create_documents_from_df(df, source_name))
            
            return docs
        except Exception as e:
            logger.error(f"Error processing Excel {excel_path}: {str(e)}")
            return []

    def _create_documents_from_df(self, df, source_name):
        documents = []
        for _, row in df.iterrows():
            query = row['query']
            answer = row['answer']
            page_content = f"Вопрос: {query}\nОтвет: {answer}"
            metadata = {
                "source": source_name,
                "doc_type": "FAQ",
                "page": ""
            }
            documents.append(LangchainDocument(page_content=page_content, metadata=metadata))
        return documents
    
    def _get_sheet_names(self, excel_path: str) -> List[str]:
        if self.excel_config.get('sheet_names'):
            return self.excel_config['sheet_names']
        return pd.ExcelFile(excel_path).sheet_names

    def _preprocess_dataframe(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        column_map = {
            self.excel_config['query_column']: 'query',
            self.excel_config['answer_column']: 'answer'
        }
        df.columns = ['query', 'answer']
        return df.pipe(self._validate_dataframe, sheet_name)

    def _validate_dataframe(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        required_columns = {'query', 'answer'}
        if missing := required_columns - set(df.columns):
            raise ValueError(f"Missing columns {missing} in sheet {sheet_name}")
        return df

class TextSplitter:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer_cache = {}

    def split_documents(self, documents: List[LangchainDocument], tokenizer_name: str) -> List[LangchainDocument]:
        if tokenizer_name not in self.tokenizer_cache:
            self.tokenizer_cache[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
            
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer_cache[tokenizer_name],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS
        )
        return text_splitter.split_documents(documents)

class VectorStore:
    def __init__(self, embedding_model_name: str, persist_dir: str, device: str = "auto"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.persist_dir = persist_dir
        self.vector_db = self._init_chroma()

    def _init_chroma(self) -> Chroma:
        try:
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_model,
                collection_name="rag_docs"
            )
        except Exception as e:
            logger.error(f"Chroma init error: {str(e)}")
            raise

    def add_documents(self, documents: List[LangchainDocument]):
        texts = [doc.page_content for doc in documents]
        metadatas = [self._sanitize_metadata(doc.metadata) for doc in documents]
        
        if not self.vector_db:
            self.vector_db = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas,
                persist_directory=self.persist_dir
            )
        else:
            self.vector_db.add_texts(texts=texts, metadatas=metadatas)

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        return {k: tuple(v) if isinstance(v, list) else v for k, v in metadata.items()}

class RAGModel:
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self, config: Dict):
        self.config = config['llm']
        self.system_prompt = config['llm'].get('system_prompt', '')
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self) -> AutoTokenizer:
        model_name = self.config['model_name']
        if model_name not in self._tokenizer_cache:
            self._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        return self._tokenizer_cache[model_name]

    def _load_model(self) -> AutoModelForCausalLM:
        model_name = self.config['model_name']
        if model_name not in self._model_cache:
            kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto",
                **({"load_in_4bit": True} if self.config.get('quantization') == "4bit" else {}),
                **({"load_in_8bit": True} if self.config.get('quantization') == "8bit" else {})
            }
            self._model_cache[model_name] = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        return self._model_cache[model_name]

    def generate_response(self, context: str, query: str) -> str:
        prompt = self._build_prompt(context, query)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config['max_new_tokens'],
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return "Извините, произошла ошибка при обработке запроса."

    def _build_prompt(self, context: str, query: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""После ответа приведи источники (лучше всего ссылки или название документа), из которых ты брал информацию, например база FAQ и другие документы!
    Тема указывается в самом конце, между @@тема@@, такой вывод очень важен. Тема может быть одной из следующих: тех проблемы, заказчик, исполнитель, общее. Другие темы брать нельзя и можно брать только одну тему, формат представления только через @@theme@@! Контекст:\n{context}\n\nВопрос: {query}\nОтвет:"""}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

class RAGSystem:
    def __init__(self, config: Union[str, Dict] = "config.yml"):
        self.cfg = ConfigLoader(config).config if isinstance(config, str) else config
        self._init_components()

    def _init_components(self):
        self.doc_processor = DocumentProcessor(
            pdf_chunk_size=self.cfg['processing']['pdf_chunk_size'],
            pdf_chunk_overlap=self.cfg['processing']['pdf_chunk_overlap'],
            excel_config=self.cfg.get('excel_processing', {})
        )
        
        self.text_splitter = TextSplitter(
            chunk_size=self.cfg['embedding']['chunk_size'],
            chunk_overlap=self.cfg['embedding']['chunk_overlap']
        )
        
        self.vector_store = VectorStore(
            embedding_model_name=self.cfg['embedding']['model_name'],
            persist_dir=self.cfg['processing']['persist_directory'],
            device=self.cfg['embedding']['device']
        )
        
        self.llm = RAGModel(self.cfg)

    def load_data(self):
        with ThreadPoolExecutor(max_workers=self.cfg['processing']['num_workers']) as executor:
            futures = []
            
            if pdf_dir := self.cfg['data_sources'].get('pdf_directory'):
                futures.extend(executor.submit(self.process_file, f) 
                    for f in glob.glob(f"{pdf_dir}/*.pdf"))
                
            if excel_path := self.cfg['data_sources'].get('excel_path'):
                futures.append(executor.submit(self.process_file, excel_path))

            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")

    def process_file(self, file_path: str):
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File {file_path} not found")
            
            if file_path.lower().endswith('.pdf'):
                docs = self.doc_processor.process_pdf(file_path)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                docs = self.doc_processor.process_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            if docs:
                split_docs = self.text_splitter.split_documents(docs, self.cfg['embedding']['model_name'])
                self.vector_store.add_documents(split_docs)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

    def query(self, question: str, top_k: int = 3) -> Tuple[str, str]:
        try:
            docs = self.vector_store.vector_db.similarity_search(question, top_k)
            context = "\n\n".join(d.page_content for d in docs)
            response = self.llm.generate_response(context, question)
            parsed_response, theme = self._parse_response(response)
            
            # Формирование блока с источниками
            sources = self._format_sources(docs)
            full_response = f"{parsed_response}\n\n{sources}"
            
            return (full_response, theme)
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return ("Ошибка обработки запроса", "unknown")

    def _format_sources(self, docs: List[LangchainDocument]) -> str:
        sources = {}
        for doc in docs:
            meta = doc.metadata
            source_type = meta.get('doc_type', 'unknown')
            source_name = meta.get('source', 'unknown')
            page = meta.get('page', '')
            
            if source_type == "PDF":
                source_info = f"[{source_name}](https://{source_name}) - страница {page}"
            elif source_type == "FAQ":
                source_info = f"[База знаний](https://{source_name})"
            else:
                source_info = source_name
                
            sources[f"{source_type}-{source_name}-{page}"] = source_info
        
        if not sources:
            return ""
            
        sources_text = "### Использованные материалы:\n" + "\n".join(
            f"📄 {info}" for info in sources.values()
        )
        return sources_text
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        try:
            # Ищем последнее вхождение темы в формате @@тема@@
            theme_match = re.search(r'@@([^@]+)@@\s*$', response)
            theme = "unknown"
            
            if theme_match:
                # Извлекаем тему и очищаем от возможных пробелов
                theme = theme_match.group(1).strip()
                # Удаляем ВСЕ упоминания темы из ответа
                cleaned_response = re.sub(r'@@[^@]+@@', '', response).strip()
            else:
                # Если тема не найдена, оставляем ответ как есть
                cleaned_response = response.strip()
            
            # Удаляем служебную информацию и reasoning
            if "Ответ:\nassistant" in cleaned_response:
                # Берем только часть после последнего "Ответ:"
                cleaned_response = cleaned_response.split("Ответ:\nassistant")[-1].strip()
            
            # Дополнительная очистка от технической информации
            cleaned_response = re.sub(
                r'\s*\[.*?\]\s*|<\/?assistant>|\|.*?\||\n|', #<\/?p> 
                ' ', 
                cleaned_response
            ).strip()
    
            return (markdown.markdown(cleaned_response), theme) # markdown.markdown
            
        except Exception as e:
            logger.error(f"Response parsing error: {str(e)}")
            return (markdown.markdown(response).replace("<p>", "").replace("</p>", "") , "FAQ")

rag = RAGSystem("config.yaml")
# rag.load_data()

if __name__ == "__main__":
    try:
        rag = RAGSystem("config.yaml")
        rag.load_data()
        
        print("Система готова к работе. Введите ваш запрос (exit для выхода):")
        while True:
            query = input(">>> ").strip()
            if query.lower() in ('exit', 'quit', 'выход'):
                break
            response, theme = rag.query(query)
            print(f"\n{response}")
            print(f"{theme}")
            
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise
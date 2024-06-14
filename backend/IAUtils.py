import urllib.parse
import numpy as np
import sklearn
import nltk 
import pprint
import sklearn.metrics
import fitz
import sys
import requests
import time
import sklearn
import pandas as pd
import nltk.corpus
import pandas
import re
import sys
import multiprocessing
import json
import urllib
import os
import init_dash_servers
import dash_bootstrap_components as dbc
import plotly.data 
import plotly.express as ply
import plotly.graph_objects as graphs
import pickle
from multiprocessing.context import BaseContext
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import KMeansSMOTE,SMOTE,SMOTEN,SVMSMOTE
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,auc
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,RandomForestClassifier,VotingClassifier,StackingClassifier,GradientBoostingClassifier
from nltk.corpus import gutenberg
from nltk import ConditionalFreqDist, FreqDist,ConditionalProbDist
from nltk.corpus import inaugural
from string import punctuation
from multiprocessing.managers import BaseManager
from bs4 import BeautifulSoup
from io import BytesIO
from dash import html,dcc,Dash,Input,Output
from plotly.graph_objects import Scatter,Bar,Line
from plotly.subplots import make_subplots
from plotly.express import scatter,bar,area,bar_polar,box,choropleth,choropleth_mapbox,line
from dash.dash_table import DataTable

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('gutenberg')
nltk.download('inaugural')
stopwords.words(['english','portuguese'])



class IAssistantRuntimeData: 
  current_url = r''
  current_site_url = ''
  current_inputs = {}
  language = 'english'
  base_url_api = 'http://localhost:8080'
  data = {}
  files_dt_output = []
  global_dataframe = pd.DataFrame()
  pre_file_dt_output = 'iaruntime_data_daframe'

#################################### INSTANCE PUBLIC CLASSES ####################
runtime_configuration = IAssistantRuntimeData()


class IAssistantRequests:
  current_extension_name = None
  code_export_file = 1234
  outfile = 'iaout_content_{}{}'
  output_filename_full = ''
  number_of_phrases_to_split_document = 20
  document_splited = []

  def set_dynamic_filename(self,extension):
     self.current_extension_name = extension
    

  def get_dynamic_filename(self,extension_name=''):
      return self.outfile.format(self.code_export_file,self.current_extension_name)
  
  def get_filename_output(self)->str:    
     return self.outfile.format(self.code_export_file)
     
  def __init__(self,code_export) -> None:
     self.code_export_file=code_export
     print('IAssistant Requests: Initilized with code option {}...'.format(code_export))

  def __init__(self) -> None:
     print('IAssistant Requests: Initilized...')


  def save_html(self,content, filename):
      self.output_filename_full = self.get_dynamic_filename(extension_name=self.current_extension_name)
      with open(self.output_filename_full, 'w', encoding='utf-8') as file:
          file.write(content)

  def save_text(self,content, filename):
      self.output_filename_full = self.get_dynamic_filename(extension_name=self.current_extension_name)
      with open(self.output_filename_full, 'w', encoding='utf-8') as file:
          file.write(content)

  def save_pdf(self,content, filename):
      self.output_filename_full = self.get_dynamic_filename(extension_name=self.current_extension_name)
      with open(self.output_filename_full, 'wb') as file:
          file.write(content)

  def handle_local_file(self,file_path,filename_output):
      # Determine the file type based on the file extension
      _, file_extension = os.path.splitext(file_path)

      self.set_dynamic_filename(file_extension)

      with open(file_path, 'rb') as file:
          content = file.read()
          
          if file_extension.lower() == '.html':
              self.save_html(content.decode('utf-8'), self.get_dynamic_filename())
          elif file_extension.lower() == '.txt':
              self.save_text(content.decode('utf-8'), self.get_dynamic_filename())
          elif file_extension.lower() == '.pdf':
              self.save_pdf(content, self.get_dynamic_filename())
          else:
              print("Unsupported file type")
  
  def get_document_splited(self,content_doc,language_config)->list:
     import re
     
     if len(re.findall("(html|csv|txt|text)",self.current_extension_name.lower()))>0:
        return get_document_splited_from_text(number_to_split_by_phrase=self.number_of_phrases_to_split_document,content_doc=content_doc,language_cfg=language_config)
     else:
        return None
        
        
  
  def download_and_save_content(self,url,filename_output,input_content_doc,number_phrases_to_split_document=20,lang_cfg=runtime_configuration.language):
      if url.startswith('file://'):
          # Handle local file
          parsed_url = urllib.parse.urlparse(url)
          local_file_path = urllib.parse.unquote(parsed_url.path)
        # Correct the file path if it starts with a leading slash on Windows
          if os.name == 'nt' and local_file_path.startswith('/'):
              local_file_path = local_file_path[1:]

          self.handle_local_file(file_path=local_file_path,filename_output=filename_output)
      else:
          # Handle remote URL
          response = requests.get(url)
          content_type = response.headers.get('Content-Type')
          
          if 'text/html' in content_type and not input_content_doc:
              self.set_dynamic_filename('.html')
              soup = BeautifulSoup(response.content, 'html.parser')
              html_content = soup.getText(strip=True,separator=' ')
              self.save_html(html_content, '{}.html'.format(filename_output))
              return self.get_document_splited(html_content,language_config=lang_cfg)
          elif 'text/plain' in content_type or 'text/csv' in content_type and not input_content_doc:
              self.set_dynamic_filename('.txt')
              text_content = response.text
              self.save_text(text_content, '{}.txt'.format(filename_output))
              return self.get_document_splited(text_content,language_config=lang_cfg)
          elif 'application/pdf' in content_type and not input_content_doc:
              self.set_dynamic_filename('.pdf')
              pdf_content = response.content
              self.save_pdf(pdf_content, '{}.pdf'.format(filename_output))
          elif  input_content_doc:
             self.set_dynamic_filename('.html')
             self.save_html(input_content_doc, '{}-inputuser.html'.format(filename_output))
             return self.get_document_splited(input_content_doc,language_config=lang_cfg)
          else:
              print("Unsupported content type")
          
      return self.output_filename_full

class IAssistantDataManager(BaseManager):
   ia_secret = 'ia'.encode(encoding='utf-8')
   server_name = ''
   share_data = {'url':'http://teste'}
   server = None

   def __init__(self):
      super().__init__(address=(self.server_name,8081),authkey=self.ia_secret)
      self.register('ia_data',callable=lambda: self.share_data)
      
      
      print('\nIAssistant Data Manager Started!!!!\n')

class IAssistantRequests:
  current_extension_name = None
  code_export_file = 1234
  outfile = 'iaout_content_{}{}'
  output_filename_full = ''
  number_of_phrases_to_split_document = 20
  document_splited = []

  def set_dynamic_filename(self,extension):
     self.current_extension_name = extension
    

  def get_dynamic_filename(self,extension_name=''):
      return self.outfile.format(self.code_export_file,self.current_extension_name)
  
  def get_filename_output(self)->str:    
     return self.outfile.format(self.code_export_file)
     
  def __init__(self,code_export) -> None:
     self.code_export_file=code_export
     print('IAssistant Requests: Initilized with code option {}...'.format(code_export))

  def __init__(self) -> None:
     print('IAssistant Requests: Initilized...')


  def save_html(self,content, filename):
      self.output_filename_full = self.get_dynamic_filename(extension_name=self.current_extension_name)
      with open(self.output_filename_full, 'w', encoding='utf-8') as file:
          file.write(content)

  def save_text(self,content, filename):
      self.output_filename_full = self.get_dynamic_filename(extension_name=self.current_extension_name)
      with open(self.output_filename_full, 'w', encoding='utf-8') as file:
          file.write(content)

  def save_pdf(self,content, filename):
      self.output_filename_full = self.get_dynamic_filename(extension_name=self.current_extension_name)
      with open(self.output_filename_full, 'wb') as file:
          file.write(content)

  def save_pdf_file_binary_content(self,content, code_content_to_filename=3333,filename='ia_pdf_binary_file_output'):
      self.output_filename_full = '{}-{}.pdf'.format(filename,code_content_to_filename)
      with open(self.output_filename_full, 'wb') as file:
          file.write(content)
      return self.output_filename_full

  def save_text_file_binary_content(self,content, code_content_to_filename=3333,filename='ia_text_binary_file_output'):
      self.output_filename_full = '{}-{}.html'.format(filename,code_content_to_filename)
      with open(self.output_filename_full, 'w',encoding='utf-8') as file:
          file.write(content)
      return self.output_filename_full

  def handle_local_file(self,file_path,filename_output):
      # Determine the file type based on the file extension
      _, file_extension = os.path.splitext(file_path)

      self.set_dynamic_filename(file_extension)

      with open(file_path, 'rb') as file:
          content = file.read()
          
          if file_extension.lower() == '.html':
              self.save_html(content.decode('utf-8'), self.get_dynamic_filename())
          elif file_extension.lower() == '.txt':
              self.save_text(content.decode('utf-8'), self.get_dynamic_filename())
          elif file_extension.lower() == '.pdf':
              self.save_pdf(content, self.get_dynamic_filename())
          else:
              print("Unsupported file type")
  
  def get_document_splited(self,content_doc,language_config)->list:
     import re
     
     if len(re.findall("(html|csv|txt|text)",self.current_extension_name.lower()))>0:
        return get_document_splited_from_text(number_to_split_by_phrase=self.number_of_phrases_to_split_document,content_doc=content_doc,language_cfg=language_config)
     else:
        return None
        

  
  def download_and_save_content(self,url,filename_output,input_content_doc,number_phrases_to_split_document=20,lang_cfg=runtime_configuration.language):
      
      url = urllib.parse.unquote(url)
      is_local_file = len(re.findall(r"(file\:[/]{2,}|^\w\:|^/\w\:)",url)) > 0

      if is_local_file:
          # Handle local file
          parsed_url = urllib.parse.urlparse(url)
          local_file_path = urllib.parse.unquote(parsed_url.path)
        # Correct the file path if it starts with a leading slash on Windows
          if os.name == 'nt' and local_file_path.startswith('/'):
              local_file_path = local_file_path[1:]

          self.handle_local_file(file_path=local_file_path,filename_output=filename_output)
      else:
          # Handle remote URL
          params_input = urllib.parse.urlparse(url).params
          response = requests.get(url,allow_redirects=True,params=params_input)
          content_type = response.headers.get('Content-Type')
          
          if 'text/html' in content_type and not input_content_doc:
              self.set_dynamic_filename('.html')
              soup = BeautifulSoup(response.content, 'html.parser')
              html_content = soup.getText(strip=True,separator=' ')
              self.save_html(html_content, '{}.html'.format(filename_output))
              return self.get_document_splited(html_content,language_config=lang_cfg)
          elif 'text/plain' in content_type or 'text/csv' in content_type and not input_content_doc:
              self.set_dynamic_filename('.txt')
              text_content = response.text
              self.save_text(text_content, '{}.txt'.format(filename_output))
              return self.get_document_splited(text_content,language_config=lang_cfg)
          elif 'application/pdf' in content_type and not input_content_doc:
              self.set_dynamic_filename('.pdf')
              pdf_content = response.content
              self.save_pdf(pdf_content, '{}.pdf'.format(filename_output))
          elif  input_content_doc:
             self.set_dynamic_filename('.html')
             self.save_html(input_content_doc, '{}-inputuser.html'.format(filename_output))
             return self.get_document_splited(input_content_doc,language_config=lang_cfg)
          else:
              print("Unsupported content type")
          
      return self.output_filename_full

  def get_local_document_from_url(self,input_url,language=runtime_configuration.language):
      ia_req = IAssistantRequests()
      output_file = ia_req.download_and_save_content(url=input_url,filename_output='iassistant_file_from_post',input_content_doc=None,lang_cfg=language)     
      return ('pdf',output_file) if type(output_file)==str else ('text',output_file)

class IAssistantFreqAnalyze:
   phrases = None
   words = None
   frequency_analyze_phrases = None
   frequency_analyze_words = None
   probality_frequency_analyze = None
   language_config = runtime_configuration.language
   input_search_from_user = None
   document_content_full = None
   lt_index_searchs = []
   lt_page_numbers = []
   lt_metrics_page = []
   lt_metrics_phrases_by_page = []
   lt_searchs = []
   lt_doc_by_page = []
   lt_doc_phrases_by_page = []
   lt_doc_urls = []

   dataframe = None
   dataframe_summary_pages = None
   __data__ = None
   max_words_by_phrase = 1
   min_size_of_phrase = 8
   def get_filter_phrases_to_process(self,input_phrases):
      return [p for p in input_phrases if len(str(p).split(" "))>self.max_words_by_phrase and len(p)>self.min_size_of_phrase]

   def process_content_clear(self,input_content,regexp_input:str=r'([^\w\s]|\d+|\b\w{1,5}\b)',replace_value_found_by_regex_to:str='')->str:
      new_content = re.sub(regexp_input, replace_value_found_by_regex_to, input_content).lower()
      
      return new_content
   

   def get_runtime_global_dataframe(self):
      list_dt = []
      for f in os.listdir('.'):
         if(f.startswith(runtime_configuration.pre_file_dt_output)): 
            with open(f,'rb') as f_dt:
               list_dt.append(pd.DataFrame(pickle.load(f_dt)).copy())
      runtime_configuration.global_dataframe = pd.concat(list_dt,axis=0)     
      print(f'IAssistant: Shape of global dataframe: columns: {len(runtime_configuration.global_dataframe.columns.tolist())} rows: {len(runtime_configuration.global_dataframe.index.values)}')
   
   def save_data_to_file(self,dt:pd.DataFrame):   
      for f in os.listdir('.'):
         if(f.startswith(runtime_configuration.pre_file_dt_output)): 
            runtime_configuration.files_dt_output.append(f)
      index_last_data_file = len(runtime_configuration.files_dt_output)
      index_new_data_file = index_last_data_file + 1
      fname_output = f'{runtime_configuration.pre_file_dt_output}{index_new_data_file}'
      
      with open(fname_output,'wb') as f:
         dt.to_pickle(f)
      runtime_configuration.files_dt_output.append(fname_output)

   def get_dataframe(self,search_items:list,precision_metrics:int=3)->pandas.DataFrame:
      if search_items:
         count_vectorizer = CountVectorizer(analyzer='word',strip_accents='unicode',lowercase=True,encoding='utf-8')
         count_vectorizer_to_phrase = CountVectorizer(analyzer='word',strip_accents='unicode',lowercase=True,encoding='utf-8')
         for indice_search,search in enumerate(search_items):
            for page_doc,doc_item in enumerate(self.document_content_full):
               if not doc_item: 
                  doc_item = 'iassistant (page invalid content)!!!!!'
               data_vector = count_vectorizer.fit_transform([doc_item,search])
               similarity_metric = cosine_similarity(X=data_vector[0],Y=data_vector[1])[0][0]
               phrases_items = sent_tokenize(doc_item,language=self.language_config)
               phrases_items = self.get_filter_phrases_to_process(input_phrases=phrases_items)
               for phrase in phrases_items:
                  vectorize_phrases = count_vectorizer_to_phrase.fit_transform([phrase,search])
                  similarity_metric_phrase = cosine_similarity(X=vectorize_phrases[0],Y=vectorize_phrases[1])[0][0]
                  self.lt_index_searchs.append(indice_search)
                  self.lt_page_numbers.append(page_doc)
                  self.lt_metrics_page.append(similarity_metric)
                  self.lt_searchs.append(search) 
                  self.lt_doc_by_page.append(doc_item)
                  self.lt_metrics_phrases_by_page.append(similarity_metric_phrase)
                  self.lt_doc_phrases_by_page.append(phrase)
                  self.lt_doc_urls.append(urllib.parse.quote(runtime_configuration.data['url']))
                  #print('process vectorizer: similarity page: {} phrase:{}'.format(similarity_metric,similarity_metric_phrase))
         self.__data__ = pd.DataFrame()

         self.dataframe = self.__data__.join([
            pd.Series(self.lt_doc_urls,name='site',dtype='category'),
            pd.Series(self.lt_page_numbers,name='page_number',dtype=np.int32),
            pd.Series(self.lt_metrics_page,name='page_metric_by_page',dtype=np.float32),
            pd.Series(self.lt_index_searchs,name='search_item',dtype=np.int32),
            pd.Series(self.lt_metrics_phrases_by_page,name='phrase_metric_to_search',dtype=np.float32),
            pd.Series(self.lt_searchs,name='search_text',dtype='category'),
            pd.Series(self.lt_doc_phrases_by_page,name='phrase_text',dtype='category'),
            pd.Series(self.lt_doc_by_page,name='page_text',dtype='category')
            ],how='outer')
         
         self.__headers__={'site':'Site Origin','page_number':'Page Number',
          'page_metric_by_page':'(%) Match of Search Phrase To Page',
          'search_item':'Indice of Search Item',
          'phrase_metric_to_search':'(%) Match Between Document Phrase and User Input Search',
          'search_text':'Text of Input Search',
          'phrase_text':'Text of Partial Phrase on Document',
          'page_text':'Page Content'
          }
         
         self.dataframe['page_metric_by_page'] = self.dataframe['page_metric_by_page'].apply(lambda x: round(x,precision_metrics))
         self.dataframe['phrase_metric_to_search'] = self.dataframe['phrase_metric_to_search'].apply(lambda x: round(x,precision_metrics))
         
         self.dataframe_summary_pages = self.dataframe.groupby(by='page_number').describe()['page_metric_by_page']
         self.dataframe_summary_pages = self.dataframe_summary_pages.apply(lambda row:[round(v,3) for v in row.values],axis=0)
         self.dataframe_summary_phrases = self.dataframe.groupby(by=['page_number','search_item']).describe()['phrase_metric_to_search']
         self.dataframe_summary_phrases = self.dataframe_summary_phrases.apply(lambda row:[round(v,3) for v in row.values],axis=0)
         print('IAssistant: Start Persist Data')
         print(f'IAssistant: Current dataframe columns: {len(self.dataframe.columns.tolist())} rows: {len(self.dataframe.index.values)}')
         
         self.save_data_to_file(self.dataframe)

         self.get_runtime_global_dataframe()
         
         print('IAssistant: End Persist Data')
         
         return self.dataframe,self.dataframe_summary_pages,self.dataframe_summary_phrases
      

   
   def __init__(self,doc_input_content,language_cfg,input_searchs_cfg) -> None:
      
      self.language_config = language_cfg
      self.input_search_from_user = input_searchs_cfg
      self.document_content_full = doc_input_content
      content = ''.join(self.document_content_full)

      cleaned_text=self.process_content_clear(input_content=content)
      
      phrases = sent_tokenize(content,language=self.language_config)
      phrases = self.get_filter_phrases_to_process(input_phrases=phrases)
      words = word_tokenize(cleaned_text,self.language_config)

      self.frequency_analyze_phrases = FreqDist(phrases)
      self.frequency_analyze_words = FreqDist(words)
      print('IAssistant: Class started : words: {}, phrases: {}, document: {}, input searchs:{}'.format(len(words),len(phrases),len(self.document_content_full),len(self.input_search_from_user)))
      self.get_dataframe(search_items=self.input_search_from_user)
   
################################################### METHOD / FUNCTIONS ############################################


def get_color(intensity):
    if not 0.0 <= intensity <= 1.0:
        return 'orange'

    if 0.0 <= intensity <= 0.1:
        # Intensidades de vermelho
        red_intensity = int(255 * (intensity / 0.1))  # Map intensity [0.0, 0.1] to [0, 255]
        #return f'rgb({red_intensity}, 3, 32)'
        return 'red'

    elif 0.1 < intensity <= 0.30:
        # Intensidades de azul
        blue_intensity = int(255 * ((intensity) / 0.3))  # Map intensity [0.2, 0.5] to [0, 255]
        #return f'rgb(6, 74, {blue_intensity})'
        return 'blue'
    elif 0.30 < intensity <= 1.0:
        # Intensidades de verde
        green_intensity = int(255 * ((intensity) / 0.5))  # Map intensity [0.5, 1.0] to [0, 255]
        #return f'rgb(45,{186+( 186*green_intensity)}, 45)'
        return 'green'

def get_color_by_metric(value_metric,min_target,mean_target):
  if value_metric <= min_target: return 'red'
  if value_metric > min_target and value_metric <= mean_target: return 'blue'
  if value_metric > mean_target: return 'green'

  return 'yellow'


def get_document_splited_from_text(number_to_split_by_phrase,content_doc,language_cfg)->list:
   new_content = []
   phrases = sent_tokenize(content_doc,language=language_cfg)
   size = len(phrases)
   items = np.arange(0,size,1)
   i_init = 0
   n_part = 20
   n_end = n_part
   iterrange = np.arange(0,size,n_part)
   for i in iterrange:
      new_content.append(phrases[i_init:n_end])
      i_init = n_end
      n_end += n_part
   
   return new_content

def get_document_content(url:str):
   doc_object = fitz.open(url)
   number_of_pages = len(doc_object)
   content = []
   for p_number in range(number_of_pages):
      page_content = doc_object.load_page(p_number)
      text_content = page_content.get_text()

      content.append(text_content)
   return content


def get_metrics_by_phrases(document_content:list,phrases_input_searchs:list,language:str='english'):
  start_time = time.strftime("%c")
  
  print('IAssistant: Start time {}'.format(start_time))
  # Download NLTK resources if not already downloaded
  full_sub_docs = []
  full_sub_docs_pages = []
  #vectorizer_count = CountVectorizer(encoding='utf-8',lowercase=True,strip_accents='unicode',analyzer='word',stop_words=language)
  cross_stopwords = stopwords.words(['english','portuguese'])
  field_tokens = []
  pages = []
  tokens_freqs = []
  for p,content in enumerate(document_content):
     cleaned_text = re.sub(r'([^\w\s]|\d+|\b\w{1,5}\b)', '', content).lower()
     tokens = [t for t in word_tokenize(cleaned_text,language=runtime_configuration.language) if t not in punctuation and t not in cross_stopwords]
     freqs = FreqDist(tokens)
     
     for tk,freq in freqs.most_common(10):
        pages.append(p)
        field_tokens.append(tk)
        tokens_freqs.append(freq)
  s_pages = pd.Series(pages,name='pages')
  s_tokens = pd.Series(field_tokens,name='token')
  s_freq = pd.Series(tokens_freqs,name='frequency')
  

  print(len(s_pages),len(s_tokens),len(s_freq))
  dt_frequencies = pd.concat([s_pages,s_tokens,s_freq],axis=1)
  print('IAssistant: end time {}'.format(time.strftime('%c')))
   
  ia_freq_analyze = IAssistantFreqAnalyze(doc_input_content=document_content,language_cfg=language,input_searchs_cfg=phrases_input_searchs)
  
  return dt_frequencies,ia_freq_analyze
  
def get_metrics_details_from_page(data:pd.DataFrame):
  pages = data['pages'].unique().tolist()
  details_metrics = {}
  for p in pages:
    filter_page = data['pages']==p
    m_max = data[filter_page]['frequency'].max()
    m_mean = data[filter_page]['frequency'].mean()
    m_median = data[filter_page]['frequency'].median()
    dt = data[filter_page][['token','frequency']]
    dt.sort_values(by='frequency',ascending=False,inplace=True)
    top_tokens = []
    for i,v in enumerate(dt.to_numpy()):
       top_tokens.append('token({}): {} : {}'.format(i,v[0],v[1]))
    
    details_metrics[p]={
      'max':m_max,
      'mean':m_mean,
      'median':m_median,
      'description':['max : {}'.format(m_max),'meam : {}'.format(m_mean),'median : {}'.format(m_median)],
      'tokens':top_tokens
    }
  return details_metrics


###################################################################################################
def get_dynamic_backend_configuration():
   import base64
   endpoint_service = f"{runtime_configuration.base_url_api}/configurations"
   data = requests.get(endpoint_service).text
   data_obj = json.loads(data)
   return data_obj

def get_dynamic_data_user_input():
   import base64
   endpoint_service = f"{runtime_configuration.base_url_api}/search/options"
   data = requests.get(endpoint_service).text
   data_obj = json.loads(data)
   runtime_configuration.data = data_obj

   return data_obj

def get_dataframe_frequencies():
    data=get_dynamic_backend_configuration()
    url= urllib.parse.unquote(data['url'])
    doc = get_document_content(url=url)
    if(len(doc)==0):
       print('*'*15,'IAssistant: Document Is Empty','*'*15)
       return False
    
    data_input = get_dynamic_data_user_input()
    input_searchs = str(data_input['search_options']).split(';')
    input_searchs = [i.strip() for i in input_searchs if len(i.strip())>0]
    dt = get_metrics_by_phrases(document_content=doc,phrases_input_searchs=input_searchs,language=runtime_configuration.language)


    print('IAssistant: End Machine Learn Process: {}'.format(time.strftime('%c')))

    return dt


########################################### GRAPHIC DESIGN ########################################

def create_data_global_gui(column_filter,field_one,field_two,type_graphic_ploty,group_op,group_items,field_values,expression):
  print(f'IAssistant: Set columns dataframe to {column_filter}, column to filter: {field_one}, field: {field_two} : expression: {expression}')

  dt = runtime_configuration.global_dataframe[column_filter]
  if str(group_op).lower()=='sim' and group_items is not None:
     dt = pd.DataFrame(dt).groupby(by=group_items,as_index=False,sort=True).mean()
  
  if expression:
     dt = dt[eval(expression)]
     
  dt_gui = DataTable(
    dt.to_dict('records'),
    sort_action="native",
    sort_mode="multi",
    column_selectable="single",
    row_selectable="multi",
    row_deletable=True,
    selected_columns=[],
    selected_rows=[],
    page_action="native",
    page_current= 0,
    page_size= 20,
    fixed_rows=dict({'headers':True,'data':0}),
    include_headers_on_copy_paste=True,
    fill_width=False,
    style_cell={
        'textAlign': 'center',
        'width': '100px',  # Set a default width for all columns
        'minWidth': '100px',
        'maxWidth': '300px',
        'marginTop': '15px'
    }
    ) 
  if type_graphic_ploty=='scatter':
    fig = scatter(x=dt[field_one],y=dt[field_two],labels=[field_one,field_two],size=dt[field_two],color=dt[field_two])
  elif type_graphic_ploty=='bar':
    fig = bar(x=dt[field_one],y=dt[field_two],labels=[field_one,field_two],color=dt[field_two])
  elif type_graphic_ploty=='barh':
    fig = bar(x=dt[field_one],y=dt[field_two],labels=[field_one,field_two],orientation='h',color=dt[field_two])
  
  return dt_gui,fig


def start_server():
  app = Dash(name=__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
  default_width = 1200
  default_height = 700

  try:
    data_frequencies,ia_analyze = get_dataframe_frequencies()
  except Exception as e:
     raise(e)

  data_frequencies.sort_values(by='frequency',ascending=False,inplace=True)
  min_target = data_frequencies['frequency'].min()
  mean_target = data_frequencies['frequency'].mean()
  max_target = data_frequencies['frequency'].max()

  color_min_target = '#e57676'
  color_mean_target = data_frequencies['frequency'].mean()
  color_max_target = data_frequencies['frequency'].max()

  details_metric_pages = get_metrics_details_from_page(data=data_frequencies)

  scatter_frequencies  = ply.scatter(
    data_frame=data_frequencies,
    x=data_frequencies['pages'],
    y=data_frequencies['frequency'],
    title='IAssistant: Frequency Analyze By Words',
    width=default_width,
    height=default_height,
    color='frequency',
    color_continuous_scale='icefire',
    trendline='ols'
    )


  data_frequencies.sort_values(by=['frequency'],ascending=False,inplace=True)

  bar_metrics = ply.bar(data_frame=data_frequencies,
    x='pages',
    y='token',
    title='IAssistant: Top Words Frequencies',
    orientation='h',
    width=default_width,
    height=default_height,
    color='frequency',color_continuous_scale='icefire'  
    )
  bar_metrics.update_layout(
    yaxis=dict(autorange='reversed')
  )



  columns_by_row = np.arange(1,1,1)
  list_of_pages = data_frequencies['pages'].unique().tolist()
  rows_of_pages = list(range(1,(len(list_of_pages))))
  pag_num = 1
  layout_pages_metrics = graphs.Layout(
    yaxis=graphs.layout.YAxis(autorange='reversed',showspikes=True),
    hoversubplots='axis',
    hovermode='x unified',
    hoverdistance=10,
    xaxis=graphs.layout.XAxis(showspikes=True),
    title='IAssistant: Frequencies By Page',
    grid=dict(rows=len(rows_of_pages),columns=1),
    width=default_width,height=200*len(rows_of_pages)
  )


  data_figs_metrics_pages=[]
  top_words_to_select = 10
  for r in rows_of_pages:
    yaxis = 'y{}'.format(r)
    xaxis = 'x'
    page_name = 'Page {}'.format(r)
    d_metrics = data_frequencies['pages']==r
    d_data = data_frequencies[d_metrics].copy()
    d_data.sort_values(by=['frequency'],inplace=True,ascending=False)
    data_figs_metrics_pages.append(
      graphs.Bar(
        x=d_data['frequency'], 
        y=d_data['token'], 
        xaxis=xaxis, 
        yaxis=yaxis, 
        name=page_name,
        orientation='h',
        hoverinfo='z+y+x+name+text',
        hovertext=['{}-Freq(%): ({}),Count:{}, Is In {} Top Words: {}'.format(token,round(ia_analyze.frequency_analyze_words.freq(token),3),ia_analyze.frequency_analyze_words[token],top_words_to_select,'sim' if token in dict(ia_analyze.frequency_analyze_words.most_common(top_words_to_select)).keys() else 'não') for token,frequency in zip(d_data['token'],d_data['frequency'])])
    )
    
  fig = graphs.Figure(data=data_figs_metrics_pages,layout=layout_pages_metrics)

  pages_options = np.array(data_frequencies['pages'].unique().tolist())
  pages_options.sort()
  pages_buttons = []
  pages_popovers =[]
  for p in pages_options:
    id = 'bt-pag{}'.format(p)
    text = 'Pag-{}'.format(p)
    summary_pages = ia_analyze.dataframe_summary_pages
    summary_phrases = ia_analyze.dataframe_summary_phrases


    summary_by_phrase = summary_phrases.xs(p,level='page_number').copy()
    summary_by_phrase.drop(labels=['count'],inplace=True,axis=1)
    
    list_details = [dbc.Badge('option({}): metric ({}): {}'.format(i,summary_by_phrase.columns[i_val],val_item),color=get_color(val_item)) for i, items in [(ind,val) for ind,val in zip(summary_by_phrase.index,summary_by_phrase.values)] for i_val,val_item in enumerate(items) if val_item>0.1]

    pages_buttons.append(dbc.Button(text,id=id,className='me-4',style={'width':'100px','margin-top':'10px','background-color':get_color(summary_pages.iloc[p,:]['max'])}))
    pages_popovers.append(dbc.Popover([
      dbc.PopoverHeader('Match(%) By Page({}):'.format(text)),
      dbc.PopoverBody(children=[dbc.Badge('{}: {}'.format(i,k),color=get_color(k)) for i,k in zip(summary_pages.iloc[p,:].index,summary_pages.iloc[p,:].values)]),
      dbc.PopoverHeader('Match(%) By Phrases({}):'.format(text)),
      dbc.PopoverBody(children=list_details.copy())
      ]
    ,target=id,
    trigger='click'))
  data_frequencies['pages'] = data_frequencies['pages']+1
  data_frequencies.sort_values(by=['frequency'],ascending=False,inplace=True)

  ia_analyze.dataframe.sort_values(by=['phrase_metric_to_search'],ascending=False,inplace=True)

  data_metrics_phrases = []


  data_metrics_phrases.append(
    graphs.Scatter(x=ia_analyze.dataframe['page_number'],y=ia_analyze.dataframe['phrase_metric_to_search'],xaxis='x',yaxis='y',name='Phrases (%)',mode='markers',marker=graphs.scatter.Marker(sizemode='diameter',cmin=0.0,cmax=1.0,cmid=0.5,size=ia_analyze.dataframe['phrase_metric_to_search']*100),hoverinfo='text',hovertext=['phrase(%) {}\npage number (%): {}\nphrase: {}'.format(phrase,pag_num,pag_metric) for phrase,pag_num,pag_metric in zip(ia_analyze.dataframe['phrase_metric_to_search'],ia_analyze.dataframe['page_metric_by_page'],ia_analyze.dataframe['phrase_text'])])
  )

  data_metrics_phrases.append(
      graphs.Scatter(x=ia_analyze.dataframe['page_number'],y=ia_analyze.dataframe['page_metric_by_page'],xaxis='x',yaxis='y2',name='Page (%)',mode='markers',marker=graphs.scatter.Marker(sizemode='diameter',cmin=0.0,cmax=1.0,cmid=0.5,size=ia_analyze.dataframe['page_metric_by_page']*100))
  )


  fig_phrases_scatter = graphs.Figure(data=data_metrics_phrases,layout=graphs.Layout(
    hoversubplots='axis',
    hovermode='y unified',
    title='IAssistant: Percent Match Between Document Content and Search Inputs Phrases',
    grid=dict(rows=2,columns=1),
    yaxis=graphs.layout.YAxis(range=[0.0,1.0],dtick=0.1,insiderange=(0.0,1.0),tickmode='auto',showspikes=True,title='Phrase (%)',spikemode='across',spikesnap='cursor'),
    xaxis=graphs.layout.XAxis(tickvals=ia_analyze.dataframe['page_number'],showspikes=True,title='Page Numbers',spikemode='across',spikesnap='cursor'),
    height=800
  ))
  data_global_dt = runtime_configuration.global_dataframe.to_dict('records')
  datable_global_metrics = DataTable(
    runtime_configuration.global_dataframe.to_dict('records'),
    sort_action="native",
    sort_mode="multi",
    column_selectable="single",
    row_selectable="multi",
    row_deletable=True,
    selected_columns=[],
    selected_rows=[],
    page_action="native",
    page_current= 0,
    page_size= 10,
    fixed_rows=dict({'headers':True,'data':0}),
    include_headers_on_copy_paste=True,
    fill_width=False,
    style_cell={
        'textAlign': 'center',
        'width': '100px',  # Set a default width for all columns
        'minWidth': '100px',
        'maxWidth': '300px',
    }
    )

  datatable_phrases_metrics = DataTable(
    ia_analyze.dataframe.to_dict('records'),
    sort_action="native",
    sort_mode="multi",
    column_selectable="single",
    row_selectable="multi",
    row_deletable=True,
    selected_columns=[],
    selected_rows=[],
    page_action="native",
    page_current= 0,
    page_size= 20,
    fixed_rows=dict({'headers':True,'data':0}),
    tooltip_header=dict(ia_analyze.__headers__),
    include_headers_on_copy_paste=True,
    fill_width=False,
    style_cell={
        'textAlign': 'center',
        'width': '100px',  # Set a default width for all columns
        'minWidth': '100px',
        'maxWidth': '300px',
    }
    )

  datatable_frequencies = DataTable(data=data_frequencies.to_dict("records"),
          columns=[{"name": i, "id": i,"presentation":"dropdown"} for i in data_frequencies.columns],
          editable=True,
          fill_width=False,
          sort_action="native",
          sort_mode="multi",
          column_selectable="single",
          row_selectable="multi",
          row_deletable=True,
          selected_columns=[],
          selected_rows=[],
          page_action="native",
          page_current= 0,
          page_size= 20,
          fixed_rows=dict({'headers':True,'data':0}),        
          include_headers_on_copy_paste=True,
          style_cell={
              'textAlign': 'center',
              'width': '200px',  # Set a default width for all columns
              'minWidth': '200px',
              'maxWidth': '300px',
          }                                 
          ) 
  tb_global_container = dbc.Container(
            children=datable_global_metrics,id="tb_global_dt"
         )
  app.layout = html.Div(
    [
      dbc.ListGroupItem(children=pages_buttons),
      dbc.Container(children=[
         dbc.Label('Name of Field:'),
         dcc.Dropdown(runtime_configuration.global_dataframe.columns.tolist(),id='ia_filter',multi=True),
         dbc.Label('Group Data:'),
         dcc.RadioItems(options=['Sim','Não'],id='ia_group_data'),
         dbc.Label('Fields To Group in Order:'),
         dcc.Dropdown(runtime_configuration.global_dataframe.columns.tolist(),id='ia_group_fields',multi=True),
         dbc.Label('Type Of Graphic:'),
         dcc.Dropdown(['scatter','bar','barh'],id='ia_type_graphic',multi=False),
         dbc.Label('Field One To Graphic(X):'),
         dcc.Dropdown(runtime_configuration.global_dataframe.columns.tolist(),id='ia_field_one',multi=False),
         dbc.Label('Field Two To Graphic:(Y)'),
         dcc.Dropdown(runtime_configuration.global_dataframe.columns.tolist(),id='ia_field_two',multi=False),
         dbc.Label('Expression To Filter:'),
         dbc.Input(placeholder='Write an expression using "dt" (dataframe pandas) and press enter...',id='ia_filter_expression',debounce=True)
         ,tb_global_container,
         dcc.Graph(id="graph_one")
      ]),
      dbc.Container(children=pages_popovers),
      dcc.Graph(id='scatter_freqs',figure=scatter_frequencies),
      dcc.Graph(id='bar_freqs',figure=bar_metrics),
      dcc.Graph(id='metrics_by_pages',figure=fig),
      dbc.CardHeader("Details Metrics Analyzes By Phrases"),
      datatable_phrases_metrics,
      dcc.Graph(id='metrics_to_match_phrases',figure=fig_phrases_scatter),
      dbc.CardHeader("Data Details of Frequency Metrics By Words"),
      datatable_frequencies
    ]
  )
  

  @app.callback(
                Output(component_id='tb_global_dt',component_property='children'),
                Output(component_id='graph_one',component_property='figure'),
                Input(component_id='ia_filter',component_property='value'),
                Input(component_id='ia_field_one',component_property='value'),
                Input(component_id='ia_field_two',component_property='value'),
                Input(component_id='ia_type_graphic',component_property='value'),
                Input(component_id='ia_group_data',component_property='value'),
                Input(component_id='ia_group_fields',component_property='value'),
                Input(component_id='ia_filter_expression',component_property='value'))
  def update_global_gui(columns_selected,field_x,field_y,type_graphic,group_options,group_fields,expression_input):
     return create_data_global_gui(column_filter=columns_selected,field_one=field_x,field_two=field_y,group_op=group_options,group_items=group_fields,type_graphic_ploty=type_graphic,expression=expression_input,field_values=None)
  
  return app



if __name__=='__main__':
   print('*'*30,'IAssistant: Module Loaded','*'*30)

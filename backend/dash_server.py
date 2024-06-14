from IAUtils import IAssistantRequests,IAssistantRuntimeData
from flask import Flask
from flask import redirect, request
from urllib import parse
from flask_cors import CORS
from client import sendQuestionGenIA
import init_dash_servers
import os
import base64
import urllib


runtime_data = IAssistantRuntimeData()
ia_download_manager = IAssistantRequests()

def start_api_server():
    app_core = Flask(import_name=__name__)
    CORS(app_core, origins='*', allow_headers='*')

    @app_core.route('/bokeh')
    def send_file_bokeh():
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        static_directory = app_core.static_folder
        arquivos = os.listdir(static_directory)
        
        return app_core.send_static_file("bokeh-3.4.1.min.js")

    @app_core.route('/metrics')
    def send_file_metrics():
        return redirect("index.html")

    @app_core.route('/graphics')
    def send_file_graphics():
        return redirect("http://localhost:8081/index.html")

    @app_core.route('/configurations')
    def get_configurations():
        import json
        ia_req = IAssistantRequests()
        url_tmp  = urllib.parse.quote(runtime_data.current_url)
        data={}
        data['url'] = url_tmp
        obj_json = json.dumps(data)
        
        return obj_json,200

    @app_core.route('/search/options')
    def get_input_data():
        import json
        obj_json = json.dumps(runtime_data.current_inputs)
        
        return obj_json,200

    @app_core.route('/api/v1/document', methods=['POST'])
    def create_document():
        import time
        data = request.json
        
        inputSearch = parse.unquote(base64.b64decode(data["inputSearchTarget"]))
        listOptionsGenIA = data['listOptionsGenIA']
        type = data["type"]
        runtime_data.current_site_url = data['url']
        app_core.logger.debug('***url to process: {}***\n\n'.format(runtime_data.current_url))
        
        
        if(type=='text'):
          contentDocument = parse.unquote(base64.b64decode(data["contentText"]).decode('utf-8'))
          binary_content = parse.unquote(base64.b64decode(data["BinaryContent"]).decode('utf-8'))
          runtime_data.current_url = ia_download_manager.save_text_file_binary_content(content=binary_content,filename='ia_html_binary_file_output',code_content_to_filename=0000)
        if(type=='pdf'):
          binary_content = base64.b64decode(data['BinaryContent'])
          contentDocument = parse.unquote(base64.b64decode(data["pdf"]).decode('utf-8'))
          runtime_data.current_url = ia_download_manager.save_pdf_file_binary_content(content=binary_content)
        
        app_core.logger.debug('***initializing dash server in app core....***')

        output = sendQuestionGenIA(inputSearch,contentDocument,number_response_options=listOptionsGenIA,creativity_degree=1.5)
        decodeData = {
            'inputSearchTarget':inputSearch,
            'search_options':output,
            'tabId':data['tabId'],
            'listOptionsGenIA': listOptionsGenIA,
            'doc':contentDocument,
            'url':runtime_data.current_site_url
            }
        runtime_data.current_inputs = decodeData.copy()

        init_dash_servers.init_dashboard_services()
        app_core.logger.debug('***initializing dash server finished!!!***')
        time.sleep(15)
        return "output.html",201

    return app_core

if __name__ == '__main__':
   app = start_api_server()
   app.run(port=8080,host='0.0.0.0')
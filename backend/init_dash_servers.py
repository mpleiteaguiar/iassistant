import sys
import subprocess

IAssistantProcessList = []

def init_dashboard_services():
    # Caminho para o script que você quer executar
    script_path = 'backend\dash_process.py'

    # Comando para executar o script com o interpretador Python
    command = [sys.executable, script_path]
    try:
       for proc in IAssistantProcessList:
           proc.terminate()
       IAssistantProcessList.clear()
    except:
        pass
        
       
    # Inicia o processo em segundo plano
    IAssistantProcessList.append(subprocess.Popen(command))
    # Output do processo para garantir que ele iniciou corretamente
    print(f"Processo iniciado com PID: {IAssistantProcessList[-1].pid}: size list: {len(IAssistantProcessList)}")

    # Opcional: você pode esperar que o processo termine se quiser.
    # process.wait()

    # Se você quiser interagir mais com o processo, você pode capturar a saída:
    # stdout, stderr = process.communicate()

    # Nota: o processo rodará em segundo plano, e você pode parar ele usando:
    # process.terminate()

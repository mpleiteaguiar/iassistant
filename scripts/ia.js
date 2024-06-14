globalThis.packageToProcessSend = {};
//globalThis.backEndEndPointServer = "https://bonefish-master-evenly.ngrok-free.app";
globalThis.backEndEndPointServer = "http://localhost:8080"
globalThis.globalThis.hiperParameterRandomize = 3
/**
 * 
 * Configure handle to input site and search
 * 
 */

console.log("IAssistant : loading libraries...");

//Message Gateway
chrome.runtime.onMessage.addListener(async (data)=>{
    console.log("IAssistent : Start Fields");
    globalThis.app_state="standby"
    setBackgroundColorWaitState()
    var { pdfjsLib } = globalThis;
    pdfjsLib.GlobalWorkerOptions.workerSrc = './pdf.worker.mjs';
    console.log(`IAssistant (PDF.js - Module build ${pdfjsLib.build})`)
    globalThis.allowSend=true
    globalThis.currentTabIdUser = null;
    globalThis.inputSite = document.getElementById("input_site");
    globalThis.listOptionsGenIA = document.getElementById("list_options_ia");
    globalThis.outputResult = document.getElementById("container_page");
    globalThis.inputSearchTarget = document.getElementById("textarea_search");
    globalThis.containerShowMetrics = document.getElementById("container_metrics");
    globalThis.contentMainDiv = document.getElementById("content_main");
    globalThis.containerGraphicsPage = document.getElementById("container_metrics");
  
    globalThis.checkbox_page = document.getElementById('show_page');
    globalThis.checkbox_graphics = document.getElementById('show_graphics');
    globalThis.checkbox_metrics = document.getElementById('show_metrics');
    globalThis.checkbox_configurations = document.getElementById('show_configurations');

    document.getElementById("list_options_ia").value=globalThis.hiperParameterRandomize;
    document.getElementById("server_address").value=globalThis.backEndEndPointServer;
    document.getElementById('list_options_ia').value = globalThis.hiperParameterRandomize;

    document.getElementById("graphs_view").src=`${globalThis.backEndEndPointServer}/graphics`;
    document.getElementById("metrics_view").src=`${globalThis.backEndEndPointServer}/metrics`;

    // Add an event listener for the 'change' event
    globalThis.checkbox_page.addEventListener('change', function(event) {
        // Check if the checkbox is checked
        if (globalThis.checkbox_page.checked) {
            container_graphics = document.getElementById("container_graphics");
            container_graphics.setAttribute("class","container-hidden");
            container_metrics = document.getElementById("container_metrics");
            container_metrics.setAttribute("class","container-hidden");
            container_configurations = document.getElementById("container_configurations");
            container_configurations.setAttribute("class","container-hidden");

            container_page = document.getElementById("container_page");
            container_page.setAttribute("class","container-visible"); 

            globalThis.checkbox_graphics.checked=false;  
            globalThis.checkbox_metrics.checked=false;   
            globalThis.checkbox_configurations.checked=false; 
        }
        
    });
    
    
    // Add an event listener for the 'change' event
    globalThis.checkbox_graphics.addEventListener('change', function(event) {
        // Check if the checkbox is checked
        if (globalThis.checkbox_graphics.checked) { 
            container_graphics = document.getElementById("container_graphics");
            container_graphics.setAttribute("class","container-visible");
            container_metrics = document.getElementById("container_metrics");
            container_metrics.setAttribute("class","container-hidden");
            container_configurations = document.getElementById("container_configurations");
            container_configurations.setAttribute("class","container-hidden");
            container_page = document.getElementById("container_page");
            container_page.setAttribute("class","container-hidden");
            
            globalThis.checkbox_page.checked=false;  
            globalThis.checkbox_metrics.checked=false;   
            globalThis.checkbox_configurations.checked=false;  
            
            document.getElementById("graphs_view").src=`${globalThis.backEndEndPointServer}/graphics`;
        }
    });
    
    // Add an event listener for the 'change' event
    globalThis.checkbox_metrics.addEventListener('change', function(event) {
        // Check if the checkbox is checked
        if (globalThis.checkbox_metrics.checked) {
            container_graphics = document.getElementById("container_graphics");
            container_graphics.setAttribute("class","container-hidden");
            container_metrics = document.getElementById("container_metrics");
            container_metrics.setAttribute("class","container-visible");
            container_configurations = document.getElementById("container_configurations");
            container_configurations.setAttribute("class","container-hidden");

            container_page = document.getElementById("container_page");
            container_page.setAttribute("class","container-hidden");  
            console.log("uncheck item");
            globalThis.checkbox_graphics.checked=false;  
            globalThis.checkbox_page.checked=false;  
            globalThis.checkbox_configurations.checked=false;     
            document.getElementById("metrics_view").src=`${globalThis.backEndEndPointServer}/metrics`;
        }
    });


    // Add an event listener for the 'change' event
    globalThis.checkbox_configurations.addEventListener('change', function(event) {
        // Check if the checkbox is checked
        if (globalThis.checkbox_configurations.checked) {
            container_graphics = document.getElementById("container_graphics");
            container_graphics.setAttribute("class","container-hidden");
            container_metrics = document.getElementById("container_metrics");
            container_metrics.setAttribute("class","container-hidden");
            container_page = document.getElementById("container_page");
            container_page.setAttribute("class","container-hidden");  

            container_configurations = document.getElementById("container_configurations");
            container_configurations.setAttribute("class","container-visible");

            console.log("uncheck item");
            globalThis.checkbox_graphics.checked=false;  
            globalThis.checkbox_page.checked=false;   
            globalThis.checkbox_metrics.checked=false    
        }});

        



        // Create a new ResizeObserver instance
        const resizeObserver = new ResizeObserver(entries => {
          // Loop through each entry in the observer
          for (let entry of entries) {
            // Log the new dimensions of the observed element
            //globalThis.subGraphicsPage.width = entry.contentRect.width-10;
            //globalThis.subGraphicsPage.height = 500;
          }
        });
        
        // Start observing the resizable div for size changes
        //resizeObserver.observe(globalThis.contentMainDiv);


        globalThis.inputSearchTarget.addEventListener("change",async (evt,obj)=>{
            var urlTarget = globalThis.inputSearchTarget.value;
            
            globalThis.packageToProcessSend.inputSearchTarget = btoa(encodeURIComponent(urlTarget));
            sendAnalyzeData(globalThis.packageToProcessSend,globalThis.containerShowMetrics);
            document.getElementById('list_options_ia').value = globalThis.hiperParameterRandomize;
            document.getElementById("graphs_view").src=`${globalThis.backEndEndPointServer}/graphics`;
            document.getElementById("metrics_view").src=`${globalThis.backEndEndPointServer}/metrics`;
        });



        chrome.windows.getCurrent().then(async (dataWindow)=>{
            url = data[dataWindow.id].url;
            globalThis.currentTabIdUser = data[dataWindow.id].tabId;
            console.log("Get Window ID");
            console.log(dataWindow.id);
            console.log(url);
            try{
                fetch(url).then((rsp)=>rsp.text().then(async (response)=>
                {
                    console.log("IAssistant: Processing type of document....");
                    if(isPDF(url)){
                        console.log(`IAssistant: PDF document Ok - ${url}`);
                        
                        // Fetch the PDF as an ArrayBuffer
                        const response = await fetch(url);
                        const arrayBuffer = await response.arrayBuffer();
                        
                        // Convert ArrayBuffer to Blob
                        const blob = new Blob([arrayBuffer], { type: 'application/pdf' });
                        file = new FileReader()
                        file.onload = function(evt){
                            content = evt.target.result
                            globalThis.packageToProcessSend.BinaryContent = btoa(content)
                        }
                        file.readAsBinaryString(blob)
                        globalThis.outputResult.innerHTML=await getPDFContent(url);
                        globalThis.packageToProcessSend.type='pdf'
                    }
                    else{
                        console.log(`IAssistant: HTML document Ok - ${url}`);
                        globalThis.outputResult.innerHTML=response;
                        globalThis.packageToProcessSend.type='text'
                        globalThis.packageToProcessSend.BinaryContent = btoa(encodeURIComponent(globalThis.outputResult.innerText));                       
                    }

                    globalThis.packageToProcessSend.inputSearchTarget=btoa(encodeURIComponent(globalThis.packageToProcessSend.inputSearchTarget));
                    globalThis.packageToProcessSend.tabId = globalThis.currentTabIdUser;
                    globalThis.packageToProcessSend.storageData = btoa(encodeURIComponent(JSON.stringify(data)));
                    globalThis.packageToProcessSend.listOptionsGenIA = globalThis.listOptionsGenIA.value;
                    globalThis.packageToProcessSend.pdf = btoa(encodeURIComponent(globalThis.outputResult.innerHTML))
                    globalThis.packageToProcessSend.contentText = btoa(encodeURIComponent(globalThis.outputResult.innerText));
                    globalThis.packageToProcessSend.contentHTML = btoa(encodeURIComponent(globalThis.outputResult.innerHTML));
                    globalThis.packageToProcessSend.url = encodeURI(data[dataWindow.id].url)
                }));
            }
            catch(e){
                console.warn(e);
            }
    
            globalThis.inputSite.value=url;
    
            /** 
            chrome.pageCapture.saveAsMHTML({tabId:globalThis.currentTabIdUser},async (data)=>{
                const pageReaderFile = new FileReader();
                
                pageReaderFile.onload=(evt)=>{
                    var contentPage = evt.target.result;
                    globalThis.currentPageContentInText = contentPage;
                };
                await pageReaderFile.readAsText(data);
            });
            */
    
        });    

        document.getElementById('list_options_ia').value = globalThis.hiperParameterRandomize;
        document.getElementById("graphs_view").src=`${globalThis.backEndEndPointServer}/graphics`;
        document.getElementById("metrics_view").src=`${globalThis.backEndEndPointServer}/metrics`;
    });
     




function iaLoadConfigurations(){
    globalThis.inputSite = document.getElementById("input_site");
    inputSite.addEventListener("IAssistantTabLoadedEvent",(data,obj)=>{
        console.log("IAssistent: Input Site Changed!!!");
        globalThis.inputSite.value=data.title;
    });
    
    console.log("ready to raise events.");
}

async function getPDFContent(urlAddress) {
    globalThis.packageToProcessSend.pdfContentPages = {};
    PDF = await pdfjsLib.getDocument(urlAddress);

    await PDF.promise.then(async function(proxyPDF) {
        proxyPDF = proxyPDF;
        numPages = proxyPDF.numPages;
        
        for (let numPage = 1; numPage <= numPages; numPage++) {
            globalThis.packageToProcessSend.pdfContentPages[numPage] = "";
            console.log(`reading page: ${numPage}`);
            
            await proxyPDF.getPage(numPage).then(async function(page) {
                await page.getTextContent().then(function(content) {
                    content.items.forEach(function(item) {
                        globalThis.packageToProcessSend.pdfContentPages[numPage] += item.str + ' ';
                    });
                });
            });
        }
    });

    console.log("Content Pages:");
    console.table(globalThis.packageToProcessSend.pdfContentPages);
    console.log("Finish - PDF Read!!!");

    var content = "";
    for(var k of Object.keys(globalThis.packageToProcessSend.pdfContentPages)){
       content +=globalThis.packageToProcessSend.pdfContentPages[k];
    }
    return content;
}



function setBackgroundColorWaitState(){

    if(globalThis.app_state==false){
        document.getElementById("msg_state").innerText="waiting..."
    }
    else{

        setInterval(()=>{
            if(globalThis.app_state==false){
                document.getElementById("msg_state").innerText="waiting..."
                document.body.style.backgroundColor='lighblue';
            }
            else if(globalThis.app_state==true){
                document.getElementById("msg_state").innerText="ready!!!!!"
                document.body.style.backgroundColor='green';
            }
            else if(globalThis.app_state=="standby"){
                document.getElementById("msg_state").innerText="standby!!!!!"
                document.body.style.backgroundColor='blue';
            }            
        },3000)
    }

}
function sendAnalyzeData(package,elementOutput){
    globalThis.app_state = false
    //const endpoint = "http://192.168.0.110:5000/api/v1/document"
    const endpoint = `${globalThis.backEndEndPointServer}/api/v1/document`
    // Data to be sent in the POST request
    // Data to be sent in the POST request
    
    // Make a POST request to the route

    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(package)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to create document');
        }
        globalThis.app_state = true
        return response.text();
    })
    .then(data => {
        try{
            fetch(data).then(contentHtml=>{
                contentHtml.text().then(content=>{
                    globalThis.app_state = true
                })
                //Get html graphics source
            });
        }
        catch(e){
            console.error(e);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function isPDF(url) {
    
    if (url.match(".pdf")) {
        console.log("The URL ends with '.pdf'.");

        return true;
    } else {
        console.log("The URL does not end with '.pdf'.");

        return false;
    }
}

console.log("IAssistant: finish - libraries loaded!!!!");
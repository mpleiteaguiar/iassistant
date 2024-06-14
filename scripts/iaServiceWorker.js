import "./uiBackEndService.js"
import "./dataServiceBackEnd.js"



console.log('IAssistant (WorkService): Loading pdf libraries....')

console.log('IAssistant (WorkService): Loading pdf libraries....')

/**
 * 
 * Define custom events
 * 
 */



globalThis.lastTabCreated = null;
globalThis.tabItem = {};
globalThis.keyItem = {};
globalThis.statusTab = "";
globalThis.waitTimePage = 3;
globalThis.iassistentListEvents = []

var { pdfjsLib } = globalThis;

// Display the suggestions after user starts typing
chrome.omnibox.onInputEntered.addListener(async (input, suggest) => {
  await chrome.omnibox.setDefaultSuggestion({
    description: 'You will send this address to IAssistent Analyze'
  });

  console.log("content pass by user:");
  console.log(input);

  globalThis.newTab = await chrome.tabs.create({ url: input });
  setTimeout(()=>{
    //console.log("IAssistent: Waiting for -> %s",input);
  },5000);
  globalThis.statusTab = globalThis.newTab.status;
  while (globalThis.statusTab!="complete") {
    setTimeout(()=>{
      //console.log("IAssistent: Waiting for -> %s",input);
    },2000);
    globalThis.newTab = await chrome.tabs.get(globalThis.newTab.id);
    globalThis.statusTab = globalThis.newTab.status;
  }
  globalThis.tabItem["windowId"]=globalThis.newTab.windowId;
  globalThis.tabItem["tabId"]=globalThis.newTab.id;
  globalThis.tabItem["url"] = globalThis.newTab.url;
  globalThis.tabItem["title"] = globalThis.newTab.title;
  globalThis.tabItem["status"] = globalThis.newTab.status;
  globalThis.tabItem["active"] = globalThis.newTab.active;
  
  globalThis.keyItem[globalThis.tabItem["windowId"]] = globalThis.tabItem;
  await chrome.storage.local.set(globalThis.keyItem);
  globalThis.lastTabCreated = await globalThis.newTab;
  console.log("New search started:",globalThis.tabItem)
  console.log(globalThis.keyItem);
  console.log(globalThis.lastTabCreated);
  console.log("IAssistent: Sending configurations data...");
  globalThis.iassistentURL = globalThis.newTab.url;

  //send data configuration
  await chrome.runtime.sendMessage(globalThis.keyItem);
  console.log("IAssistent: data sended!!!");
  //try{
    //await chrome.scripting.executeScript({target : {tabId : globalThis.newTab.id},files : ["./scripts/ia.js"]}).then((result)=>console.log(result));
  //}
  //catch(e){
    //console.error(e);
  //}
  
  console.log("IAssistent: Finish injection!!!");
  //Inject scripts
  //await chrome.scripting.executeScript({target : {tabId : 880544896},files : ["./scripts/ia.js"]}).then((result)=>console.log(result));
  /**
   * 
   * This is for change suggestion descriptions options
   * 
   */
  //const { apiSuggestions } = await chrome.storage.local.get('apiSuggestions');
  //const suggestions = apiSuggestions.map((api) => {
  // return { content: api, description: `Open chrome.${api} API` };
  //});
});


async function getTabConfigurations(tab){
  globalThis.lastTab = await chrome.tabs.get(tab);
  console.log("Update Last tab:")
  console.log(lastTab);
}

function getUserInputFromURL(text,objSearch){
    console.log("User send new input to IAssistent:")
    console.log(text);
    console.log(objSearch);
}


// Save default API suggestions
chrome.runtime.onInstalled.addListener(({ reason }) => {
  if (reason === 'install') {
    chrome.storage.local.set({
      apiSuggestions: ['tabs', 'storage', 'scripting']
    });
  }
});



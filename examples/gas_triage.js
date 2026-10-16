const JEV_MCP_WEBHOOK_URL = 'TEMPLATE';
const CONTEXT_ID = 'default_mailbox'; // Change this if you have multiple mailboxes

// When the smart engine fails, the webhook answers 503 (with Retry-After) for the first
// attempts and the unconfirmed result with 200 on the last. Keep this equal to the server's
// MAX_DEGRADED_ATTEMPTS; the script sends its attempt number in the X-Triage-Attempt header.
const MAX_DEGRADED_ATTEMPTS = 3;
const RETRY_PAUSE_MS = 5000;

// The webhook key is read from Script Properties (Project Settings > Script Properties,
// property JEV_MCP_API_KEY), so it is not part of this source or its version history.
function getApiKey_() {
  return PropertiesService.getScriptProperties().getProperty('JEV_MCP_API_KEY') || '';
}

function processUnreadEmails() {
  if (JEV_MCP_WEBHOOK_URL === 'TEMPLATE') {
    Logger.log("Please set your JEV_MCP_WEBHOOK_URL at the top of the script.");
    return;
  }

  // Find unread emails in the inbox
  const threads = GmailApp.search('is:unread in:inbox', 0, 10);
  
  if (threads.length === 0) {
    Logger.log("No unread emails found.");
    return;
  }
  
  for (let i = 0; i < threads.length; i++) {
    const thread = threads[i];
    const messages = thread.getMessages();
    const lastMessage = messages[messages.length - 1];
    
    // Only process unread messages
    if (!lastMessage.isUnread()) {
      continue;
    }
    
    const subject = lastMessage.getSubject();
    const sender = lastMessage.getFrom();
    const body = lastMessage.getPlainBody();
    const messageId = lastMessage.getId();
    
    Logger.log(`Processing: ${subject} from ${sender}`);
    
    const payload = {
      context_id: CONTEXT_ID,
      id: messageId,
      subject: subject,
      sender: sender,
      body: body
    };
    
    const options = {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify(payload),
      muteHttpExceptions: true
    };
    
    // Add authentication header if the Script Property is set
    const apiKey = getApiKey_();
    const headers = {};
    if (apiKey.length > 0) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }
    
    try {
      let response;
      for (let attempt = 1; attempt <= MAX_DEGRADED_ATTEMPTS; attempt++) {
        headers['X-Triage-Attempt'] = String(attempt);
        options.headers = headers;
        response = UrlFetchApp.fetch(JEV_MCP_WEBHOOK_URL, options);
        // 503 is a degraded triage (or a stopped engine) worth retrying after a pause.
        if (response.getResponseCode() !== 503 || attempt === MAX_DEGRADED_ATTEMPTS) {
          break;
        }
        Logger.log(`Attempt ${attempt} answered 503; retrying.`);
        Utilities.sleep(RETRY_PAUSE_MS);
      }
      const responseCode = response.getResponseCode();
      const responseBody = response.getContentText();
      
      if (responseCode >= 200 && responseCode < 300) {
        Logger.log(`Successfully routed message to Jev-MCP: ${responseCode}`);
        // Mark as read so we don't process it again in the next tick
        lastMessage.markRead();
      } else {
        Logger.log(`Failed to route message: HTTP ${responseCode}. Response: ${responseBody}`);
      }
    } catch (e) {
      Logger.log(`Error calling webhook: ${e.toString()}`);
    }
  }
}

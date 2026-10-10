const JEV_MCP_WEBHOOK_URL = 'TEMPLATE';
const JEV_MCP_API_KEY = 'TEMPLATE';
const CONTEXT_ID = 'default_mailbox'; // Change this if you have multiple mailboxes

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
    
    // Add authentication header if API key is provided
    if (JEV_MCP_API_KEY !== 'TEMPLATE' && JEV_MCP_API_KEY.length > 0) {
      options.headers = {
        'Authorization': `Bearer ${JEV_MCP_API_KEY}`
      };
    }
    
    try {
      const response = UrlFetchApp.fetch(JEV_MCP_WEBHOOK_URL, options);
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

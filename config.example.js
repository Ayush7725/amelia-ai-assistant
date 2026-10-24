// Copy to config.js (git-ignored) and fill in real values.
window.AMELIA_CONFIG = {
  // Human-agent voice handoff over SIP (WebRTC)
  sip: {
    uri: "sip:USER@YOUR_SIP_HOST",
    server: "wss://YOUR_SIP_HOST:8089/ws",
    domain: "YOUR_SIP_HOST",
    username: "USER",
    password: "CHANGE_ME"
  },
  // Contact-centre / dialer endpoints used for tickets and chat sync
  dialer: {
    ticketUrl: "https://YOUR_DIALER_HOST/api/generate_ticket.php",
    nonAgentApiUrl: "https://YOUR_DIALER_HOST/api/non_agent_api.php",
    apiUser: "CHANGE_ME",
    apiPass: "CHANGE_ME"
  }
};

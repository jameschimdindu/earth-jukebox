// Vercel function: gives the browser a short-lived ElevenLabs credential for Juke.
// The API key stays on the server. Set ELEVENLABS_API_KEY in Vercel, Project, Settings, Environment Variables.
// GET /api/token?type=webrtc  -> { token }      (voice, WebRTC)
// GET /api/token?type=ws      -> { signedUrl }  (text only, WebSocket)
const AGENT_ID = 'agent_4301m7cxc5g6ew78t8x4egjwrb4r';

module.exports = async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  const key = process.env.ELEVENLABS_API_KEY;
  if (!key) {
    res.status(503).json({ error: 'not_configured' });
    return;
  }
  const type = req.query && req.query.type === 'ws' ? 'ws' : 'webrtc';
  const url = type === 'ws'
    ? `https://api.elevenlabs.io/v1/convai/conversation/get-signed-url?agent_id=${AGENT_ID}`
    : `https://api.elevenlabs.io/v1/convai/conversation/token?agent_id=${AGENT_ID}`;
  try {
    const r = await fetch(url, { headers: { 'xi-api-key': key } });
    if (!r.ok) {
      res.status(502).json({ error: 'upstream', status: r.status });
      return;
    }
    const j = await r.json();
    res.status(200).json(type === 'ws' ? { signedUrl: j.signed_url } : { token: j.token });
  } catch (e) {
    res.status(502).json({ error: 'network' });
  }
};

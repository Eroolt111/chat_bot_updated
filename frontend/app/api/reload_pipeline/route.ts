const FLASK_URL = process.env.FLASK_URL ?? 'http://localhost:5000';

export async function POST() {
  const res = await fetch(`${FLASK_URL}/api/reload_pipeline`, { method: 'POST' });
  const data = await res.json();
  return Response.json(data, { status: res.status });
}

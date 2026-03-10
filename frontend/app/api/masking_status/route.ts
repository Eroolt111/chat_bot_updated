const FLASK_URL = process.env.FLASK_URL ?? 'http://localhost:5000';

export async function GET() {
  const res = await fetch(`${FLASK_URL}/api/masking_status`);
  const data = await res.json();
  return Response.json(data, { status: res.status });
}

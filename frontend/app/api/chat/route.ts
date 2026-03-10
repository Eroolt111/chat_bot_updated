import { NextRequest } from 'next/server';

const FLASK_URL = process.env.FLASK_URL ?? 'http://localhost:5000';

export async function POST(req: NextRequest) {
  const body = await req.json();
  const res = await fetch(`${FLASK_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return Response.json(data, { status: res.status });
}

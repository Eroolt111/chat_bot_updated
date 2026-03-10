import { NextRequest } from 'next/server';

const FLASK_URL = process.env.FLASK_URL ?? 'http://localhost:5000';

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ session_id: string }> }
) {
  const { session_id } = await params;
  const res = await fetch(`${FLASK_URL}/api/session_stats/${session_id}`);
  const data = await res.json();
  return Response.json(data, { status: res.status });
}

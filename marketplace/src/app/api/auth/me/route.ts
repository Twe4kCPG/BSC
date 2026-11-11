import { NextResponse } from 'next/server';
import jwt from 'jsonwebtoken';
import { cookies } from 'next/headers';
import { prisma } from '@/lib/prisma';

const JWT_SECRET = process.env.JWT_SECRET || 'super-secret-jwt-key';

export async function GET(request: Request) {
  try {
    const cookieStore = cookies();
    const token = cookieStore.get('token')?.value;

    if (!token) {
      return NextResponse.json({ authenticated: false }, { status: 200 });
    }

    try {
      const decoded = jwt.verify(token, JWT_SECRET) as { userId: number; username: string };

      const user = await prisma.user.findUnique({
        where: { id: decoded.userId },
        select: { id: true, username: true, karma: true }
      });

      if (!user) {
        return NextResponse.json({ authenticated: false }, { status: 200 });
      }

      return NextResponse.json({ authenticated: true, user }, { status: 200 });
    } catch (err) {
      return NextResponse.json({ authenticated: false }, { status: 200 });
    }
  } catch (error) {
    console.error('Auth check error:', error);
    return NextResponse.json({ error: 'Something went wrong' }, { status: 500 });
  }
}

export async function DELETE(request: Request) {
    const response = NextResponse.json({ message: 'Logged out successfully' }, { status: 200 });
    response.cookies.delete('token');
    return response;
}
import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import jwt from 'jsonwebtoken';
import { cookies } from 'next/headers';

const JWT_SECRET = process.env.JWT_SECRET || 'super-secret-jwt-key';

export async function POST(request: Request, { params }: { params: { id: string } }) {
  try {
    const cookieStore = cookies();
    const token = cookieStore.get('token')?.value;

    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    try {
      jwt.verify(token, JWT_SECRET); // just checking validity
    } catch (err) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 });
    }

    const { action } = await request.json();

    if (action !== 'upvote' && action !== 'downvote') {
      return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
    }

    const userId = parseInt(params.id, 10);
    const userToUpdate = await prisma.user.findUnique({
      where: { id: userId },
    });

    if (!userToUpdate) {
        return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    const karmaChange = action === 'upvote' ? 1 : -1;
    let newKarma = userToUpdate.karma + karmaChange;

    if (newKarma > 100) newKarma = 100;
    if (newKarma < -100) newKarma = -100;

    const updatedUser = await prisma.user.update({
        where: { id: userId },
        data: { karma: newKarma }
    });

    return NextResponse.json({ message: 'Karma updated', karma: updatedUser.karma }, { status: 200 });

  } catch (error) {
    console.error('Update karma error:', error);
    return NextResponse.json({ error: 'Something went wrong' }, { status: 500 });
  }
}

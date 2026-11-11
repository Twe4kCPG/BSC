import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import jwt from 'jsonwebtoken';
import { cookies } from 'next/headers';

const JWT_SECRET = process.env.JWT_SECRET || 'super-secret-jwt-key';

export async function POST(request: Request) {
  try {
    const cookieStore = cookies();
    const token = cookieStore.get('token')?.value;

    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    let decoded: any;
    try {
      decoded = jwt.verify(token, JWT_SECRET);
    } catch (err) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 });
    }

    const { title, description, price, category, images } = await request.json();

    if (!title || !description || price === undefined || !category) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    const imageArray = images ? images.split(',').map((u: string) => u.trim()).filter(Boolean) : [];

    const offer = await prisma.offer.create({
      data: {
        title,
        description,
        price: parseFloat(price),
        category,
        images: imageArray,
        userId: decoded.userId,
      },
    });

    return NextResponse.json(offer, { status: 201 });
  } catch (error) {
    console.error('Create offer error:', error);
    return NextResponse.json({ error: 'Something went wrong' }, { status: 500 });
  }
}

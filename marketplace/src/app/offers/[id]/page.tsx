import { prisma } from '@/lib/prisma';
import { notFound } from 'next/navigation';
import Link from 'next/link';

export default async function OfferDetails({ params }: { params: { id: string } }) {
  const offerId = parseInt(params.id, 10);
  const offer = await prisma.offer.findUnique({
    where: { id: offerId },
    include: { user: true },
  });

  if (!offer) {
    notFound();
  }

  return (
    <div className="max-w-2xl mx-auto mt-10">
      <h1 className="text-3xl font-bold mb-2">{offer.title}</h1>
      <p className="text-gray-500 mb-4">{offer.category}</p>
      <div className="bg-gray-100 p-6 rounded mb-6">
        <p className="text-2xl font-bold mb-4">€{offer.price.toFixed(2)}</p>
        <p className="whitespace-pre-wrap">{offer.description}</p>
      </div>

      <div className="border-t pt-6">
        <h2 className="text-xl font-semibold mb-2">Seller Information</h2>
        <p>Username: {offer.user.username}</p>
        <p>Trust Score (Karma): {offer.user.karma} / 100</p>
        <Link href={`/profile/${offer.user.id}`} className="text-blue-600 hover:underline mt-2 inline-block">
          View Seller Profile
        </Link>
      </div>
    </div>
  );
}

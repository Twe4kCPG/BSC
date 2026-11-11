import { prisma } from '@/lib/prisma';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import VoteButtons from './VoteButtons';

export default async function Profile({ params }: { params: { id: string } }) {
  const userId = parseInt(params.id, 10);
  const user = await prisma.user.findUnique({
    where: { id: userId },
    include: { offers: true },
  });

  if (!user) {
    notFound();
  }

  return (
    <div className="max-w-4xl mx-auto mt-10">
      <div className="bg-gray-100 p-6 rounded mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold mb-2">{user.username}&apos;s Profile</h1>
          <p className="text-lg">Trust Score (Karma): <span className="font-bold">{user.karma}</span> / 100</p>
        </div>
        <VoteButtons userId={user.id.toString()} />
      </div>

      <h2 className="text-2xl font-semibold mb-4">Offers by {user.username}</h2>
      {user.offers.length === 0 ? (
        <p>This user has no active offers.</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {user.offers.map((offer) => (
            <div key={offer.id} className="border rounded p-4 shadow-sm">
              <h3 className="text-xl font-semibold mb-2">{offer.title}</h3>
              <p className="text-gray-600 mb-2">{offer.category}</p>
              <p className="text-lg font-bold mb-4">€{offer.price.toFixed(2)}</p>
              <Link
                href={`/offers/${offer.id}`}
                className="bg-blue-600 text-white px-4 py-2 rounded inline-block"
              >
                View Offer
              </Link>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

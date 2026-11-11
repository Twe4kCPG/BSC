import { prisma } from '@/lib/prisma';
import Link from 'next/link';

export const dynamic = 'force-dynamic';

export default async function Home() {
  const offers = await prisma.offer.findMany({
    orderBy: { createdAt: 'desc' },
  });

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Latest Offers</h1>
      {offers.length === 0 ? (
        <p>No offers available right now.</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {offers.map((offer) => (
            <div key={offer.id} className="border rounded p-4 shadow-sm">
              <h2 className="text-xl font-semibold mb-2">{offer.title}</h2>
              <p className="text-gray-600 mb-2">{offer.category}</p>
              <p className="text-lg font-bold mb-4">€{offer.price.toFixed(2)}</p>
              <Link
                href={`/offers/${offer.id}`}
                className="bg-blue-600 text-white px-4 py-2 rounded inline-block"
              >
                View Details
              </Link>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

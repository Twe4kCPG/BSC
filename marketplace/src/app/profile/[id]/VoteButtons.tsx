'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';

export default function VoteButtons({ userId }: { userId: string }) {
  const router = useRouter();
  const [error, setError] = useState('');

  const handleVote = async (action: 'upvote' | 'downvote') => {
    setError('');
    const res = await fetch(`/api/users/${userId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });

    if (res.ok) {
      router.refresh(); // refresh the server component to show new karma
    } else {
      const data = await res.json();
      setError(data.error || 'Failed to vote');
    }
  };

  return (
    <div className="flex flex-col items-end gap-2">
      <div className="flex gap-2">
        <button
          onClick={() => handleVote('upvote')}
          className="bg-green-500 text-white px-3 py-1 rounded"
        >
          👍 +1
        </button>
        <button
          onClick={() => handleVote('downvote')}
          className="bg-red-500 text-white px-3 py-1 rounded"
        >
          👎 -1
        </button>
      </div>
      {error && <p className="text-red-500 text-sm">{error}</p>}
    </div>
  );
}

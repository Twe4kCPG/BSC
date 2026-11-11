'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

export default function Navbar() {
  const router = useRouter();
  const [user, setUser] = useState<{ id: string; username: string } | null>(null);

  useEffect(() => {
    fetch('/api/auth/me')
      .then((res) => res.json())
      .then((data) => {
        if (data.authenticated) {
          setUser(data.user);
        } else {
          setUser(null);
        }
      })
      .catch((err) => console.error(err));
  }, []);

  const handleLogout = async () => {
    await fetch('/api/auth/me', { method: 'DELETE' });
    setUser(null);
    router.push('/');
    router.refresh();
  };

  return (
    <nav className="bg-blue-600 text-white p-4">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" className="text-xl font-bold">
          EuroMarket
        </Link>
        <div className="flex gap-4 items-center">
          {user ? (
            <>
              <Link href="/offers/create" className="hover:underline">
                Create Offer
              </Link>
              <Link href={`/profile/${user.id}`} className="hover:underline">
                Profile ({user.username})
              </Link>
              <button onClick={handleLogout} className="hover:underline">
                Logout
              </button>
            </>
          ) : (
            <>
              <Link href="/login" className="hover:underline">
                Login
              </Link>
              <Link href="/register" className="bg-white text-blue-600 px-3 py-1 rounded">
                Register
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}

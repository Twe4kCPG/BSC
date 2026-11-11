import { test, expect } from '@playwright/test';

test.describe('Marketplace MVP Flow', () => {
  const testUser = {
    username: `testuser_${Date.now()}`,
    password: 'password123'
  };

  test('can register, login, and create an offer', async ({ page }) => {
    // 1. Register
    await page.goto('/register');
    await page.fill('input[placeholder="Username"]', testUser.username);
    await page.fill('input[placeholder="Password"]', testUser.password);
    await page.click('button[type="submit"]');

    // 2. Login
    await page.waitForURL('/login');
    await page.fill('input[placeholder="Username"]', testUser.username);
    await page.fill('input[placeholder="Password"]', testUser.password);
    await page.click('button[type="submit"]');

    // 3. Verify Navbar shows logged in state
    await page.waitForURL('/');
    await expect(page.locator('nav')).toContainText('Logout');

    // 4. Create Offer
    await page.click('text=Create Offer');
    await page.waitForURL('/offers/create');
    const offerTitle = `Vintage Laptop ${Date.now()}`;
    await page.fill('input[placeholder="Title"]', offerTitle);
    await page.fill('input[placeholder="Category (e.g. Electronics)"]', 'Electronics');
    await page.fill('input[placeholder="Price (€)"]', '250');
    await page.fill('textarea[placeholder="Description"]', 'Works great, slightly used.');
    await page.click('button[type="submit"]');

    // 5. Check Home Page for new offer
    await page.waitForURL('/');
    await expect(page.locator(`text=${offerTitle}`).first()).toBeVisible();

    // 6. View Offer Details
    await page.locator(`text=${offerTitle}`).first().locator('..').locator('text=View Details').click();
    await expect(page.locator(`text=${offerTitle}`).first()).toBeVisible();
    await expect(page.locator(`text=${testUser.username}`)).toBeVisible();
  });
});

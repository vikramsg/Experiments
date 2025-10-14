# Next.js Learning Guide: From Basics to Production

## Table of Contents
1. [What is Next.js?](#what-is-nextjs)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Creating Your First Next.js Project](#creating-your-first-nextjs-project)
5. [Project Structure Explained](#project-structure-explained)
6. [Pages and Routing](#pages-and-routing)
7. [Components and React Fundamentals](#components-and-react-fundamentals)
8. [Styling Options](#styling-options)
9. [Data Fetching Strategies](#data-fetching-strategies)
10. [API Routes](#api-routes)

---

## What is Next.js?

### Overview
Next.js is a **React framework** built by Vercel that provides a complete solution for building modern web applications. Think of it as React with superpowers.

### Why Next.js vs Plain React?

| Plain React | Next.js |
|-------------|---------|
| Client-side only | Server-side rendering (SSR) + Client-side |
| Manual routing setup | File-based routing (automatic) |
| No built-in SEO optimization | SEO-friendly out of the box |
| Requires separate backend | API routes built-in |
| Manual code splitting | Automatic code splitting |
| Complex configuration | Zero-config to start |

### Key Features
- **Server-Side Rendering (SSR)**: Pages are rendered on the server, improving performance and SEO
- **Static Site Generation (SSG)**: Generate HTML at build time for blazing-fast sites
- **File-based Routing**: Create pages by simply adding files to the `pages` directory
- **API Routes**: Build your backend API within the same project
- **Image Optimization**: Automatic image optimization with the `<Image>` component
- **TypeScript Support**: First-class TypeScript support out of the box

### When to Use Next.js?
- Building SEO-critical websites (blogs, e-commerce, marketing sites)
- Applications requiring fast initial page loads
- Full-stack applications with frontend and backend in one codebase
- Content-heavy sites that benefit from static generation

---

## Prerequisites

Before starting with Next.js, you should have:

### Required Knowledge
1. **TypeScript Fundamentals**
   - Type annotations: `const name: string = 'value'`
   - Interfaces and types: `interface User { name: string }`
   - Arrow functions: `const fn = (): void => {}`
   - Destructuring: `const { name } = obj`
   - Spread operator: `...array`
   - Promises and async/await
   - Modules (import/export)
   - Generics: `Array<T>`

2. **React Basics**
   - JSX syntax
   - Components (functional components)
   - Props and state
   - Hooks (useState, useEffect)
   - Event handling

3. **HTML & CSS**
   - Basic HTML structure
   - CSS selectors and properties
   - Flexbox and Grid (helpful but not required)

### Required Software
- **Node.js**: Version 18.17 or later
- **Package Manager**: npm (comes with Node.js), yarn, or pnpm
- **Code Editor**: VS Code recommended
- **Terminal**: Basic command line knowledge

### Checking Your Setup
```bash
# Check Node.js version
node --version
# Should output: v18.17.0 or higher

# Check npm version
npm --version
# Should output: 9.0.0 or higher
```

---

## Environment Setup

### Installing Node.js
If you don't have Node.js installed:

**macOS/Linux:**
```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install --lts
nvm use --lts
```

**Windows:**
- Download from [nodejs.org](https://nodejs.org/)
- Run the installer
- Restart your terminal

### Choosing a Package Manager

**npm** (comes with Node.js):
```bash
npm --version
```

## Creating Your First Next.js Project

### Method 1: Using create-next-app (Recommended)

This is the official and easiest way to start a Next.js project:

```bash
# Create a new Next.js app
npx create-next-app@latest my-first-nextjs-app

# You'll be asked several questions:
# ✔ Would you like to use TypeScript? › No / Yes
# ✔ Would you like to use ESLint? › No / Yes
# ✔ Would you like to use Tailwind CSS? › No / Yes
# ✔ Would you like to use `src/` directory? › No / Yes
# ✔ Would you like to use App Router? (recommended) › No / Yes
# ✔ Would you like to use Turbopack? (recommended) › No / Yes
# ✔ Would you like to customize the default import alias? › No / Yes
```

**Recommendations for beginners:**
- TypeScript: Yes 
- ESLint: Yes (catches errors)
- Tailwind CSS: Yes (easy styling)
- `src/` directory: No (simpler structure)
- App Router: No (start with Pages Router - simpler to learn)
- Import alias: No (default is fine)

```bash
# Navigate into your project
cd nextjs-app

# Start the development server
npm run dev
```

### Manual Setup for reference 

To understand what create-next-app does, let's create a Next.js project manually:

```bash
# Create project directory
mkdir my-nextjs-app
cd my-nextjs-app

# Initialize npm
npm init -y

# Install Next.js, React, React DOM, and TypeScript dependencies
npm install next react react-dom
npm install --save-dev typescript @types/react @types/node

# Create pages directory
mkdir pages
```

Create a `tsconfig.json` file (Next.js will populate it on first run):
```json
{}
```

Update `package.json` to add scripts:
```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  }
}
```

Create your first page `pages/index.tsx`:
```typescript
export default function Home(): JSX.Element {
  return <h1>Hello Next.js!</h1>
}
```

### Understanding the Dev Server

When you run `npm run dev`:
- Development server starts at `http://localhost:3000`
- Hot Module Replacement (HMR) is enabled (changes reflect instantly)
- Error overlay shows helpful error messages
- Fast Refresh preserves component state during edits

**Open your browser and visit:** `http://localhost:3000`

You should see your Next.js app running!

---

## Project Structure Explained

Let's explore the structure of a Next.js project created with create-next-app:

```
my-first-nextjs-app/
├── node_modules/          # Dependencies (don't touch)
├── pages/                 # Your pages and routes
│   ├── _app.tsx          # Custom App component (wraps all pages)
│   ├── _document.tsx     # Custom Document (HTML structure)
│   ├── index.tsx         # Home page (route: /)
│   └── api/              # API routes
│       └── hello.ts      # API endpoint
├── public/               # Static files (images, fonts)
│   ├── favicon.ico
│   └── vercel.svg
├── styles/               # CSS files
│   ├── globals.css       # Global styles
│   └── Home.module.css   # Component-specific styles
├── .eslintrc.json        # ESLint configuration
├── .gitignore            # Git ignore rules
├── next.config.js        # Next.js configuration (can be .ts)
├── package.json          # Project metadata and dependencies
├── tsconfig.json         # TypeScript configuration
└── README.md             # Project documentation
```

### Key Directories and Files

#### 1. `pages/` Directory
**Most important!** This is where your application routes live.

- `pages/index.tsx` → `/` (home page)
- `pages/about.tsx` → `/about`
- `pages/blog/index.tsx` → `/blog`
- `pages/blog/[id].tsx` → `/blog/:id` (dynamic route)

**Every file in `pages/` becomes a route automatically!**

#### 2. `public/` Directory
Static assets that are served from the root:

```
public/logo.png → Access as: /logo.png
public/images/hero.jpg → Access as: /images/hero.jpg
```

Example:
```jsx
<img src="/logo.png" alt="Logo" />
```

#### 3. `styles/` Directory
CSS files for styling:

- **globals.css**: Styles applied to entire app
- **Module.css**: Component-scoped styles (CSS Modules)

#### 4. `pages/_app.tsx`
Special file that wraps all pages. Used for:
- Global layouts
- Global state management
- Adding global CSS
- Persisting state between page changes

```typescript
import type { AppProps } from 'next/app'
import '../styles/globals.css'

export default function MyApp({ Component, pageProps }: AppProps): JSX.Element {
  return <Component {...pageProps} />
}
```

#### 5. `next.config.js`
Configure Next.js behavior:

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['example.com'],
  },
}

module.exports = nextConfig
```

### Hidden Directories (Created on First Build)

#### `.next/` Directory
Build output and cache (automatically generated, don't edit)

#### `node_modules/`
All installed dependencies

---

## Pages and Routing

### File-Based Routing Concept

Unlike React Router where you define routes in code, Next.js uses the **file system as the routing system**.

**Rule:** Every `.ts` or `.tsx` file in `pages/` becomes a route.

### Basic Routes

#### Example 1: Simple Pages
```
pages/
├── index.tsx         → /
├── about.tsx         → /about
├── contact.tsx       → /contact
└── pricing.tsx       → /pricing
```

**pages/about.tsx:**
```typescript
export default function About(): JSX.Element {
  return (
    <div>
      <h1>About Us</h1>
      <p>Welcome to our Next.js site!</p>
    </div>
  )
}
```

Visit: `http://localhost:3000/about`

#### Example 2: Nested Routes
```
pages/
├── index.tsx                   → /
├── blog/
│   ├── index.tsx              → /blog
│   ├── first-post.tsx         → /blog/first-post
│   └── second-post.tsx        → /blog/second-post
└── products/
    ├── index.tsx              → /products
    └── category/
        └── electronics.tsx    → /products/category/electronics
```

**pages/blog/index.tsx:**
```typescript
export default function Blog(): JSX.Element {
  return <h1>Blog Home</h1>
}
```

**pages/blog/first-post.tsx:**
```typescript
export default function FirstPost(): JSX.Element {
  return <h1>My First Post</h1>
}
```

### Dynamic Routes

Use square brackets `[param]` for dynamic route segments.

#### Example 3: Dynamic Blog Posts
```
pages/
└── blog/
    ├── index.tsx             → /blog
    └── [slug].tsx            → /blog/:slug (any slug)
```

**pages/blog/[slug].tsx:**
```typescript
import { useRouter } from 'next/router'

export default function BlogPost(): JSX.Element {
  const router = useRouter()
  const { slug } = router.query

  return (
    <div>
      <h1>Blog Post: {slug}</h1>
      <p>You are viewing the post with slug: {slug}</p>
    </div>
  )
}
```

Now these URLs work:
- `/blog/hello-world` → slug = "hello-world"
- `/blog/nextjs-tutorial` → slug = "nextjs-tutorial"
- `/blog/any-url-here` → slug = "any-url-here"

#### Example 4: Multiple Dynamic Segments
```
pages/
└── products/
    └── [category]/
        └── [id].tsx          → /products/:category/:id
```

**pages/products/[category]/[id].tsx:**
```typescript
import { useRouter } from 'next/router'

export default function Product(): JSX.Element {
  const router = useRouter()
  const { category, id } = router.query

  return (
    <div>
      <h1>Product {id}</h1>
      <p>Category: {category}</p>
    </div>
  )
}
```

URLs:
- `/products/electronics/123` → category="electronics", id="123"
- `/products/clothing/456` → category="clothing", id="456"

### Catch-All Routes

Use `[...param]` to catch all segments.

```
pages/
└── docs/
    └── [...slug].tsx         → /docs/* (any depth)
```

**pages/docs/[...slug].tsx:**
```typescript
import { useRouter } from 'next/router'

export default function Docs(): JSX.Element {
  const router = useRouter()
  const { slug } = router.query

  // slug is an array of path segments
  return (
    <div>
      <h1>Documentation</h1>
      <p>Path segments: {JSON.stringify(slug)}</p>
    </div>
  )
}
```

URLs:
- `/docs/intro` → slug = ["intro"]
- `/docs/api/auth` → slug = ["api", "auth"]
- `/docs/api/auth/login` → slug = ["api", "auth", "login"]

### Navigation Between Pages

#### Using the Link Component

**Never use `<a>` tags for internal links!** Use Next.js `<Link>`:

```typescript
import Link from 'next/link'

export default function Navigation(): JSX.Element {
  return (
    <nav>
      <Link href="/">Home</Link>
      <Link href="/about">About</Link>
      <Link href="/blog">Blog</Link>
      <Link href="/blog/hello-world">First Post</Link>
    </nav>
  )
}
```

**Why Link is better:**
- Client-side navigation (no full page reload)
- Prefetches pages in the background
- Faster navigation
- Maintains scroll position

#### Using the Router Programmatically

```typescript
import { useRouter } from 'next/router'

export default function LoginButton(): JSX.Element {
  const router = useRouter()

  const handleLogin = async (): Promise<void> => {
    // Perform login logic
    const success = await loginUser()

    if (success) {
      // Navigate to dashboard
      router.push('/dashboard')
    }
  }

  return <button onClick={handleLogin}>Login</button>
}

// Mock function for example
async function loginUser(): Promise<boolean> {
  return true
}
```

**Router methods:**
- `router.push('/path')` - Navigate to a page
- `router.replace('/path')` - Navigate without adding to history
- `router.back()` - Go back one page
- `router.reload()` - Reload current page
- `router.prefetch('/path')` - Prefetch a page

### 404 Page

Create `pages/404.tsx` for custom 404 pages:

```typescript
export default function Custom404(): JSX.Element {
  return (
    <div>
      <h1>404 - Page Not Found</h1>
      <p>The page you're looking for doesn't exist.</p>
    </div>
  )
}
```

---

## Components and React Fundamentals

### React Components in Next.js

Next.js is built on React, so all React concepts apply. Let's review the fundamentals.

### Creating Components

**Recommended structure:**
```
my-nextjs-app/
├── components/
│   ├── Header.tsx
│   ├── Footer.tsx
│   └── Button.tsx
└── pages/
    └── index.tsx
```

**components/Header.tsx:**
```typescript
import Link from 'next/link'

export default function Header(): JSX.Element {
  return (
    <header>
      <h1>My Website</h1>
      <nav>
        <Link href="/">Home</Link>
        <Link href="/about">About</Link>
      </nav>
    </header>
  )
}
```

**Using the component:**
```typescript
// pages/index.tsx
import Header from '../components/Header'

export default function Home(): JSX.Element {
  return (
    <div>
      <Header />
      <main>
        <h1>Welcome to my site</h1>
      </main>
    </div>
  )
}
```

### Props (Passing Data to Components)

Props allow you to pass data from parent to child components.

**components/Button.tsx:**
```typescript
interface ButtonProps {
  text: string
  color: string
  onClick: () => void
}

export default function Button({ text, color, onClick }: ButtonProps): JSX.Element {
  return (
    <button
      style={{ backgroundColor: color }}
      onClick={onClick}
    >
      {text}
    </button>
  )
}
```

**Using the Button component:**
```typescript
import Button from '../components/Button'

export default function Home(): JSX.Element {
  const handleClick = (): void => {
    alert('Button clicked!')
  }

  return (
    <div>
      <Button text="Click me" color="blue" onClick={handleClick} />
      <Button text="Delete" color="red" onClick={() => alert('Deleted!')} />
    </div>
  )
}
```

### State Management with useState

State allows components to remember data between renders.

```typescript
import { useState } from 'react'

export default function Counter(): JSX.Element {
  // Declare state variable
  const [count, setCount] = useState<number>(0)

  return (
    <div>
      <p>You clicked {count} times</p>
      <button onClick={() => setCount(count + 1)}>
        Click me
      </button>
      <button onClick={() => setCount(0)}>
        Reset
      </button>
    </div>
  )
}
```

**How useState works:**
1. `useState<number>(0)` initializes state with value `0` and type annotation
2. Returns an array: `[currentValue, updateFunction]`
3. Calling `setCount(newValue)` updates state and re-renders component

### Side Effects with useEffect

useEffect runs code after the component renders (for side effects like API calls).

```typescript
import { useState, useEffect } from 'react'

interface User {
  name: string
  email: string
}

export default function UserProfile(): JSX.Element {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState<boolean>(true)

  useEffect(() => {
    // This runs after the component mounts
    fetch('https://api.example.com/user')
      .then(response => response.json())
      .then((data: User) => {
        setUser(data)
        setLoading(false)
      })
  }, []) // Empty array = run only once on mount

  if (loading) return <p>Loading...</p>
  if (!user) return <p>No user found</p>

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  )
}
```

**useEffect dependency array:**
```typescript
useEffect(() => {
  // Runs on every render
})

useEffect(() => {
  // Runs only once on mount
}, [])

useEffect(() => {
  // Runs when 'count' changes
}, [count])

useEffect(() => {
  // Runs when 'count' or 'name' changes
}, [count, name])
```

### Component Composition

Build complex UIs by composing small components.

**components/Layout.tsx:**
```typescript
import { ReactNode } from 'react'
import Header from './Header'
import Footer from './Footer'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps): JSX.Element {
  return (
    <div>
      <Header />
      <main style={{ minHeight: '80vh' }}>
        {children}
      </main>
      <Footer />
    </div>
  )
}
```

**pages/_app.tsx (apply layout to all pages):**
```typescript
import type { AppProps } from 'next/app'
import Layout from '../components/Layout'
import '../styles/globals.css'

export default function MyApp({ Component, pageProps }: AppProps): JSX.Element {
  return (
    <Layout>
      <Component {...pageProps} />
    </Layout>
  )
}
```

Now every page automatically has Header and Footer!

### Conditional Rendering

```javascript
export default function Greeting({ isLoggedIn, username }) {
  // Method 1: if/else
  if (isLoggedIn) {
    return <h1>Welcome back, {username}!</h1>
  } else {
    return <h1>Please log in</h1>
  }

  // Method 2: ternary operator
  return (
    <h1>
      {isLoggedIn ? `Welcome back, ${username}!` : 'Please log in'}
    </h1>
  )

  // Method 3: && operator (for showing/hiding)
  return (
    <div>
      {isLoggedIn && <h1>Welcome back, {username}!</h1>}
      {!isLoggedIn && <h1>Please log in</h1>}
    </div>
  )
}
```

### Rendering Lists

```javascript
export default function BlogList() {
  const posts = [
    { id: 1, title: 'First Post', author: 'John' },
    { id: 2, title: 'Second Post', author: 'Jane' },
    { id: 3, title: 'Third Post', author: 'Bob' },
  ]

  return (
    <ul>
      {posts.map(post => (
        <li key={post.id}>
          <h2>{post.title}</h2>
          <p>By {post.author}</p>
        </li>
      ))}
    </ul>
  )
}
```

**Important:** Always provide a unique `key` prop when rendering lists!

---

## Styling Options

Next.js supports multiple styling approaches. Let's explore each one.

### 1. Global CSS

**styles/globals.css:**
```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background-color: #f5f5f5;
}

a {
  color: #0070f3;
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}
```

**Import in pages/_app.js:**
```javascript
import '../styles/globals.css'

function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />
}

export default MyApp
```

**Note:** Global CSS can ONLY be imported in `_app.js`, not in individual pages or components.

### 2. CSS Modules (Recommended)

CSS Modules scope styles to a specific component, preventing conflicts.

**components/Button.module.css:**
```css
.button {
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
}

.primary {
  background-color: #0070f3;
  color: white;
}

.secondary {
  background-color: #eaeaea;
  color: black;
}

.button:hover {
  opacity: 0.8;
}
```

**components/Button.js:**
```javascript
import styles from './Button.module.css'

export default function Button({ text, variant = 'primary' }) {
  return (
    <button className={`${styles.button} ${styles[variant]}`}>
      {text}
    </button>
  )
}
```

**Why CSS Modules?**
- Styles are scoped to the component (no naming conflicts)
- Class names are automatically unique
- You can use simple class names without worrying about collisions
- Better than inline styles (supports pseudo-classes, media queries)

### 3. Inline Styles

```javascript
export default function Box() {
  const boxStyle = {
    backgroundColor: 'lightblue',
    padding: '20px',
    borderRadius: '10px',
    marginBottom: '10px',
  }

  return (
    <div style={boxStyle}>
      <p style={{ color: 'darkblue', fontSize: '18px' }}>
        This is a styled box
      </p>
    </div>
  )
}
```

**When to use inline styles:**
- Dynamic styles based on props or state
- Simple one-off styles
- Quick prototyping

**Limitations:**
- No pseudo-classes (`:hover`, `:focus`)
- No media queries
- Can't use CSS preprocessor features

### 4. Tailwind CSS (Most Popular)

Tailwind is a utility-first CSS framework. Instead of writing CSS, you apply pre-built classes.

**If you selected Tailwind during setup, it's already configured!**

**Example with Tailwind:**
```javascript
export default function Card({ title, description }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <h2 className="text-2xl font-bold text-gray-800 mb-2">
        {title}
      </h2>
      <p className="text-gray-600">
        {description}
      </p>
      <button className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
        Read More
      </button>
    </div>
  )
}
```

**Tailwind classes explained:**
- `bg-white` → background-color: white
- `rounded-lg` → border-radius: 0.5rem
- `shadow-md` → box-shadow: medium
- `p-6` → padding: 1.5rem
- `text-2xl` → font-size: 1.5rem
- `font-bold` → font-weight: 700
- `hover:shadow-lg` → larger shadow on hover
- `transition-shadow` → smooth shadow transition

**Why Tailwind?**
- No need to name CSS classes
- Consistent design system built-in
- Highly customizable
- Smaller bundle size (unused styles are purged)
- Fast development once you learn the classes

**Customizing Tailwind (tailwind.config.js):**
```javascript
module.exports = {
  content: [
    './pages/**/*.{js,jsx}',
    './components/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          light: '#3fbaeb',
          DEFAULT: '#0fa9e6',
          dark: '#0c87b8',
        },
      },
    },
  },
  plugins: [],
}
```

Now use: `bg-brand`, `text-brand-dark`, etc.

### 5. Styled JSX (Next.js Built-in)

Next.js comes with styled-jsx for component-scoped CSS-in-JS.

```javascript
export default function StyledComponent() {
  return (
    <div>
      <h1>Styled with styled-jsx</h1>
      <p>This is scoped to this component only</p>

      <style jsx>{`
        h1 {
          color: #0070f3;
          font-size: 2rem;
        }

        p {
          color: #666;
          line-height: 1.5;
        }

        h1:hover {
          color: #0051cc;
        }
      `}</style>
    </div>
  )
}
```

**For global styled-jsx:**
```javascript
<style jsx global>{`
  body {
    margin: 0;
  }
`}</style>
```

### Styling Recommendations

**For beginners:**
- Start with **CSS Modules** (simple, scoped, familiar CSS syntax)
- Try **Tailwind CSS** once comfortable (faster development)

**For production:**
- **Tailwind CSS** for most projects (fast, maintainable)
- **CSS Modules** for component libraries
- **Styled-components** or **Emotion** for complex theming needs

---

## Data Fetching Strategies

Next.js provides multiple ways to fetch data. Each method serves different use cases.

### Understanding Rendering Methods

#### 1. Client-Side Rendering (CSR)
- Data fetched in the browser using `useEffect`
- Good for user-specific data
- Not SEO-friendly

#### 2. Server-Side Rendering (SSR)
- Data fetched on the server for each request
- Fresh data on every page load
- SEO-friendly
- Uses `getServerSideProps`

#### 3. Static Site Generation (SSG)
- Data fetched at build time
- HTML pre-generated
- Blazing fast, SEO-friendly
- Uses `getStaticProps`

#### 4. Incremental Static Regeneration (ISR)
- Static pages that update in the background
- Best of both worlds
- Uses `getStaticProps` with `revalidate`

### 1. Client-Side Data Fetching

Fetch data in the browser using React hooks.

```javascript
import { useState, useEffect } from 'react'

export default function Users() {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('https://jsonplaceholder.typicode.com/users')
      .then(response => {
        if (!response.ok) throw new Error('Failed to fetch')
        return response.json()
      })
      .then(data => {
        setUsers(data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  if (loading) return <p>Loading users...</p>
  if (error) return <p>Error: {error}</p>

  return (
    <div>
      <h1>Users</h1>
      <ul>
        {users.map(user => (
          <li key={user.id}>{user.name}</li>
        ))}
      </ul>
    </div>
  )
}
```

**When to use:**
- User-specific data (dashboard, profile)
- Data that changes frequently
- Data that doesn't need SEO

### 2. Server-Side Rendering with getServerSideProps

Fetch data on the server on every request.

```javascript
// pages/posts.js
export default function Posts({ posts }) {
  return (
    <div>
      <h1>Blog Posts</h1>
      <ul>
        {posts.map(post => (
          <li key={post.id}>
            <h2>{post.title}</h2>
            <p>{post.body}</p>
          </li>
        ))}
      </ul>
    </div>
  )
}

// This runs on the SERVER on every request
export async function getServerSideProps() {
  const res = await fetch('https://jsonplaceholder.typicode.com/posts')
  const posts = await res.json()

  return {
    props: {
      posts, // Passed to the component as props
    },
  }
}
```

**How it works:**
1. User requests `/posts`
2. Next.js runs `getServerSideProps` on the server
3. Fetches data from API
4. Renders page with data
5. Sends HTML to browser

**When to use:**
- Data changes frequently
- Need fresh data on every request
- SEO-critical pages
- User-specific content that needs SEO

**With dynamic routes:**
```javascript
// pages/user/[id].js
export default function UserProfile({ user }) {
  return (
    <div>
      <h1>{user.name}</h1>
      <p>Email: {user.email}</p>
      <p>Phone: {user.phone}</p>
    </div>
  )
}

export async function getServerSideProps(context) {
  const { id } = context.params // Get dynamic route parameter
  const res = await fetch(`https://jsonplaceholder.typicode.com/users/${id}`)
  const user = await res.json()

  // Handle 404
  if (!user.id) {
    return {
      notFound: true, // Shows 404 page
    }
  }

  return {
    props: { user },
  }
}
```

### 3. Static Site Generation with getStaticProps

Generate static HTML at build time.

```javascript
// pages/about.js
export default function About({ company }) {
  return (
    <div>
      <h1>About {company.name}</h1>
      <p>{company.description}</p>
      <p>Founded: {company.founded}</p>
    </div>
  )
}

// This runs at BUILD TIME (once)
export async function getStaticProps() {
  const res = await fetch('https://api.example.com/company')
  const company = await res.json()

  return {
    props: {
      company,
    },
  }
}
```

**How it works:**
1. During `npm run build`, Next.js calls `getStaticProps`
2. Generates static HTML with the data
3. On request, serves pre-built HTML (super fast!)
4. No API calls happen on page load

**When to use:**
- Content doesn't change often
- Blog posts, documentation, marketing pages
- Maximum performance needed
- SEO-critical content

### 4. Static Generation with Dynamic Routes

For dynamic routes like `/blog/[slug]`, you need both `getStaticProps` and `getStaticPaths`.

```javascript
// pages/blog/[slug].js
export default function BlogPost({ post }) {
  return (
    <article>
      <h1>{post.title}</h1>
      <p>{post.content}</p>
      <small>By {post.author}</small>
    </article>
  )
}

// Tell Next.js which paths to pre-generate
export async function getStaticPaths() {
  // Fetch list of all blog posts
  const res = await fetch('https://api.example.com/posts')
  const posts = await res.json()

  // Generate paths for each post
  const paths = posts.map(post => ({
    params: { slug: post.slug },
  }))

  return {
    paths, // Array of paths to pre-generate
    fallback: false, // 404 for non-existent paths
  }
}

// Fetch data for each path
export async function getStaticProps({ params }) {
  const res = await fetch(`https://api.example.com/posts/${params.slug}`)
  const post = await res.json()

  return {
    props: { post },
  }
}
```

**How it works:**
1. `getStaticPaths` returns list of all possible slugs
2. For each slug, `getStaticProps` fetches data
3. Next.js generates HTML for all combinations
4. Result: All blog pages are pre-built!

**Fallback options:**

```javascript
return {
  paths,
  fallback: false, // 404 for non-generated paths
}

return {
  paths,
  fallback: true, // Generate on-demand, show loading state first
}

return {
  paths,
  fallback: 'blocking', // Generate on-demand, wait before showing page
}
```

### 5. Incremental Static Regeneration (ISR)

Static pages that update in the background.

```javascript
export default function Products({ products, timestamp }) {
  return (
    <div>
      <h1>Products</h1>
      <p>Last updated: {new Date(timestamp).toLocaleString()}</p>
      <ul>
        {products.map(product => (
          <li key={product.id}>{product.name}</li>
        ))}
      </ul>
    </div>
  )
}

export async function getStaticProps() {
  const res = await fetch('https://api.example.com/products')
  const products = await res.json()

  return {
    props: {
      products,
      timestamp: Date.now(),
    },
    revalidate: 60, // Regenerate page every 60 seconds
  }
}
```

**How ISR works:**
1. First request: Serves static HTML (built at build time)
2. After 60 seconds: Next request triggers background regeneration
3. While regenerating: Still serves old static page (fast!)
4. Once done: Swaps in new static page
5. Repeat

**Benefits:**
- Fast like static pages
- Fresh data without rebuilding entire site
- No downtime during updates

**When to use:**
- E-commerce product pages
- Blog with frequent updates
- News sites
- Any content that changes periodically

### Quick Comparison Table

| Method | When | Speed | SEO | Fresh Data |
|--------|------|-------|-----|------------|
| Client-Side | User data, frequent updates | Slow initial | ❌ | ✅ Always |
| SSR | Need fresh data every time | Medium | ✅ | ✅ Always |
| SSG | Static content | ⚡ Fastest | ✅ | ❌ Build time only |
| ISR | Periodic updates | ⚡ Fast | ✅ | ✅ Background |

---

## API Routes

Next.js allows you to create API endpoints in the same project as your frontend.

### Creating API Routes

**File location:** `pages/api/` directory

**Routing:**
- `pages/api/hello.js` → `/api/hello`
- `pages/api/users/[id].js` → `/api/users/:id`
- `pages/api/posts/index.js` → `/api/posts`

### Basic API Route

**pages/api/hello.js:**
```javascript
// This function runs on the server
export default function handler(req, res) {
  // req = request object
  // res = response object

  res.status(200).json({
    message: 'Hello from Next.js API!',
    timestamp: new Date().toISOString()
  })
}
```

**Test it:**
```bash
curl http://localhost:3000/api/hello
```

**Access from frontend:**
```javascript
export default function HomePage() {
  const [data, setData] = useState(null)

  useEffect(() => {
    fetch('/api/hello')
      .then(res => res.json())
      .then(data => setData(data))
  }, [])

  return <div>{data && <p>{data.message}</p>}</div>
}
```

### Handling Different HTTP Methods

```javascript
// pages/api/users.js
export default function handler(req, res) {
  const { method } = req

  switch (method) {
    case 'GET':
      // Handle GET request
      res.status(200).json({ users: ['Alice', 'Bob', 'Charlie'] })
      break

    case 'POST':
      // Handle POST request
      const { name } = req.body
      res.status(201).json({ message: `User ${name} created` })
      break

    case 'PUT':
      // Handle PUT request
      res.status(200).json({ message: 'User updated' })
      break

    case 'DELETE':
      // Handle DELETE request
      res.status(200).json({ message: 'User deleted' })
      break

    default:
      res.setHeader('Allow', ['GET', 'POST', 'PUT', 'DELETE'])
      res.status(405).end(`Method ${method} Not Allowed`)
  }
}
```

### Dynamic API Routes

**pages/api/users/[id].js:**
```javascript
export default function handler(req, res) {
  const { id } = req.query // Get dynamic parameter

  // Simulate database lookup
  const users = {
    '1': { id: 1, name: 'Alice', email: 'alice@example.com' },
    '2': { id: 2, name: 'Bob', email: 'bob@example.com' },
  }

  const user = users[id]

  if (!user) {
    return res.status(404).json({ error: 'User not found' })
  }

  res.status(200).json(user)
}
```

**Usage:**
- GET `/api/users/1` → Returns Alice
- GET `/api/users/2` → Returns Bob
- GET `/api/users/999` → Returns 404

### Request and Response Objects

#### Request (req) Properties:
```javascript
export default function handler(req, res) {
  // HTTP method
  console.log(req.method) // 'GET', 'POST', etc.

  // Query parameters (?name=value)
  console.log(req.query) // { name: 'value' }

  // Request body (for POST/PUT)
  console.log(req.body) // parsed JSON body

  // Cookies
  console.log(req.cookies) // { token: 'abc123' }

  // Headers
  console.log(req.headers) // { 'content-type': 'application/json' }
}
```

#### Response (res) Methods:
```javascript
// Send JSON
res.status(200).json({ data: 'value' })

// Send text
res.status(200).send('Plain text response')

// Set status code
res.status(404)

// Set headers
res.setHeader('Content-Type', 'application/json')

// Redirect
res.redirect(307, '/new-url')

// End response
res.end()
```

### Example: Simple Todo API

**pages/api/todos.js:**
```javascript
// In-memory storage (use a database in production)
let todos = [
  { id: 1, text: 'Learn Next.js', completed: false },
  { id: 2, text: 'Build an app', completed: false },
]

export default function handler(req, res) {
  const { method } = req

  if (method === 'GET') {
    // Return all todos
    return res.status(200).json(todos)
  }

  if (method === 'POST') {
    // Create new todo
    const { text } = req.body
    const newTodo = {
      id: todos.length + 1,
      text,
      completed: false,
    }
    todos.push(newTodo)
    return res.status(201).json(newTodo)
  }

  res.status(405).json({ error: 'Method not allowed' })
}
```

**pages/api/todos/[id].js:**
```javascript
let todos = [
  { id: 1, text: 'Learn Next.js', completed: false },
  { id: 2, text: 'Build an app', completed: false },
]

export default function handler(req, res) {
  const { method } = req
  const { id } = req.query
  const todoId = parseInt(id)

  if (method === 'PUT') {
    // Update todo
    const { completed } = req.body
    const todo = todos.find(t => t.id === todoId)

    if (!todo) {
      return res.status(404).json({ error: 'Todo not found' })
    }

    todo.completed = completed
    return res.status(200).json(todo)
  }

  if (method === 'DELETE') {
    // Delete todo
    todos = todos.filter(t => t.id !== todoId)
    return res.status(200).json({ message: 'Todo deleted' })
  }

  res.status(405).json({ error: 'Method not allowed' })
}
```

**Frontend component to use the API:**
```javascript
import { useState, useEffect } from 'react'

export default function TodoApp() {
  const [todos, setTodos] = useState([])
  const [newTodo, setNewTodo] = useState('')

  // Fetch todos
  useEffect(() => {
    fetch('/api/todos')
      .then(res => res.json())
      .then(data => setTodos(data))
  }, [])

  // Add todo
  const addTodo = async () => {
    const res = await fetch('/api/todos', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: newTodo }),
    })
    const todo = await res.json()
    setTodos([...todos, todo])
    setNewTodo('')
  }

  // Toggle todo
  const toggleTodo = async (id, completed) => {
    await fetch(`/api/todos/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ completed: !completed }),
    })
    setTodos(todos.map(t =>
      t.id === id ? { ...t, completed: !completed } : t
    ))
  }

  // Delete todo
  const deleteTodo = async (id) => {
    await fetch(`/api/todos/${id}`, { method: 'DELETE' })
    setTodos(todos.filter(t => t.id !== id))
  }

  return (
    <div>
      <h1>Todo List</h1>

      <input
        value={newTodo}
        onChange={e => setNewTodo(e.target.value)}
        placeholder="New todo..."
      />
      <button onClick={addTodo}>Add</button>

      <ul>
        {todos.map(todo => (
          <li key={todo.id}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => toggleTodo(todo.id, todo.completed)}
            />
            <span style={{
              textDecoration: todo.completed ? 'line-through' : 'none'
            }}>
              {todo.text}
            </span>
            <button onClick={() => deleteTodo(todo.id)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  )
}
```

### Connecting to a Database

Example with a simple JSON file (use a real database in production):

**lib/db.js:**
```javascript
import fs from 'fs'
import path from 'path'

const dbPath = path.join(process.cwd(), 'data.json')

export function getDB() {
  const data = fs.readFileSync(dbPath, 'utf-8')
  return JSON.parse(data)
}

export function saveDB(data) {
  fs.writeFileSync(dbPath, JSON.stringify(data, null, 2))
}
```

**pages/api/posts.js:**
```javascript
import { getDB, saveDB } from '../../lib/db'

export default function handler(req, res) {
  const db = getDB()

  if (req.method === 'GET') {
    res.status(200).json(db.posts)
  }

  if (req.method === 'POST') {
    const newPost = {
      id: db.posts.length + 1,
      ...req.body,
      createdAt: new Date().toISOString(),
    }
    db.posts.push(newPost)
    saveDB(db)
    res.status(201).json(newPost)
  }
}
```

### Environment Variables

Store sensitive data like API keys in environment variables.

**Create `.env.local`:**
```
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
API_KEY=your-secret-api-key
NEXT_PUBLIC_API_URL=https://api.example.com
```

**Important:**
- Variables prefixed with `NEXT_PUBLIC_` are exposed to the browser
- Other variables are only available on the server

**Usage in API routes:**
```javascript
export default function handler(req, res) {
  // Only available on server
  const apiKey = process.env.API_KEY

  // Available in browser too
  const publicUrl = process.env.NEXT_PUBLIC_API_URL

  res.status(200).json({ apiKey })
}
```

### Error Handling

```javascript
export default async function handler(req, res) {
  try {
    const response = await fetch('https://api.example.com/data')

    if (!response.ok) {
      throw new Error('External API failed')
    }

    const data = await response.json()
    res.status(200).json(data)
  } catch (error) {
    console.error('API Error:', error)
    res.status(500).json({
      error: 'Internal Server Error',
      message: error.message
    })
  }
}
```

### CORS (Cross-Origin Requests)

If your API needs to be accessed from other domains:

```javascript
export default function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

  // Handle preflight request
  if (req.method === 'OPTIONS') {
    return res.status(200).end()
  }

  // Your API logic
  res.status(200).json({ message: 'CORS enabled' })
}
```

---

## Summary and Next Steps

Congratulations! You've learned the fundamentals of Next.js:

1. What Next.js is and why to use it
2. Setting up your development environment
3. Creating and understanding project structure
4. File-based routing and navigation
5. React components, props, and state
6. Multiple styling approaches
7. Data fetching strategies (CSR, SSR, SSG, ISR)
8. Building API routes

### What's Next?

Continue learning with these topics:

1. **Image Optimization** - Using next/image
2. **Authentication** - Implementing user login
3. **Database Integration** - PostgreSQL, MongoDB, Prisma
4. **Form Handling** - React Hook Form, validation
5. **State Management** - Context API, Zustand, Redux
6. **Testing** - Jest, React Testing Library
7. **Deployment** - Vercel, AWS, Docker
8. **Performance** - Code splitting, lazy loading
9. **App Router** - New routing system (Next.js 13+)
10. **TypeScript** - Type safety for larger projects

### Practice Projects

Build these to solidify your learning:

1. **Personal Blog** - Practice SSG, markdown, routing
2. **Todo App** - Practice state, API routes, CRUD
3. **E-commerce Store** - Practice dynamic routes, ISR, images
4. **Dashboard** - Practice SSR, data visualization, auth
5. **Social Media Clone** - Full-stack with database, auth, real-time

### Resources

- **Official Docs:** https://nextjs.org/docs
- **Learn Next.js:** https://nextjs.org/learn
- **Next.js Examples:** https://github.com/vercel/next.js/tree/canary/examples
- **Community:** https://github.com/vercel/next.js/discussions

Happy coding!

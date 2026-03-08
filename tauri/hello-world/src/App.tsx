import './App.css'

function App() {
  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">Tauri + Vite + React + TypeScript</p>
        <h1>Hello World</h1>
        <p className="lede">
          The desktop shell is wired up and ready for local development.
        </p>
      </section>

      <section className="panel">
        <div>
          <h2>What is connected</h2>
          <ul>
            <li>Vite serves the frontend during development.</li>
            <li>React + TypeScript render the app UI.</li>
            <li>Tauri packages the web app as a native desktop window.</li>
          </ul>
        </div>

        <div className="commands">
          <h2>Useful commands</h2>
          <code>npm run dev</code>
          <code>npm run tauri dev</code>
          <code>npm run tauri build</code>
        </div>
      </section>
    </main>
  )
}

export default App

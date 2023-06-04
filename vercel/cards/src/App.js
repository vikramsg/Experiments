import './App.css';
import Home from './components/Home';
import TopBar from './components/TopBar';


const App = () => {
  return (
    <div>
      <TopBar />
      <Home />
      {/* Rest of your page content goes here */}
    </div>
  );
};

export default App;
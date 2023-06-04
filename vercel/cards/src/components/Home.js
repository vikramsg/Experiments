import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import { Link } from 'react-router-dom';

const Home = () => {
    const cards = [
        {
            title: "Hamburg",
            text: "Find out all destinations you can take from Hamburg using only your 49 Euro ticket.",
            link: "/origin/hamburg",
        },
        {
            title: "Berlin",
            text: "Coming soon!",
            link: "/origin/berlin",
        },
        {
            title: "Munich",
            text: "Coming soon!",
            link: "/origin/munich",
        },
        {
            title: "Frankfurt",
            text: "Coming soon!",
            link: "/origin/frankfurt",
        },
        {
            title: "Cologne",
            text: "Coming soon!",
            link: "/origin/cologne",
        },
        {
            title: "Stuttgart",
            text: "Coming soon!",
            link: "/origin/stuttgart",
        }
    ];

    return (
        <Container className="d-flex justify-content-center mt-4">
            <Row xs={1} md={2} lg={2} className="g-4">
                {cards.map((card, index) => (
                    <Col key={index}>
                        <Link to={card.link} className="text-decoration-none">
                            <Card className="h-100">
                                <Card.Body>
                                    <Card.Title>{card.title}</Card.Title>
                                    <Card.Text>{card.text}</Card.Text>
                                </Card.Body>
                            </Card>
                        </Link>
                    </Col>
                ))}
            </Row>
        </Container>
    );
};

export default Home;

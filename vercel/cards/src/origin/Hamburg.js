import React, { useEffect, useState } from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import { formatDuration, intervalToDuration } from 'date-fns';

import cardsData from '../data/hamburg_destinations.json';

const Hamburg = () => {
    const [cards, setCards] = useState([]);

    useEffect(() => {
        setCards(cardsData.cards);
    }, []);

    return (
        <Container className="d-flex justify-content-center mt-4">
            <Row xs={1} md={2} lg={2} className="g-4">
                {cards.map((card) => (
                    <Col key={card.city}>
                        <Card className="h-100">
                            <Card.Body>
                                <Card.Title>{card.city}</Card.Title>
                                <Card.Text>{formatDuration(intervalToDuration({ start: 0, end: card.journey_time * 1000 }))}</Card.Text>
                            </Card.Body>
                        </Card>
                    </Col>
                ))}
            </Row>
        </Container>
    );
};

export default Hamburg;
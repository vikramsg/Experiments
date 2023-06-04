import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';

const Hamburg = () => {
    // Generate an array of card data
    const cardData = Array.from({ length: 100 }, (_, index) => ({
        id: index + 1,
        title: `Card ${index + 1}`,
        description: `This is card number ${index + 1}`,
    }));

    return (
        <Container className="mt-4">
            <Row xs={1} md={2} lg={3} className="g-4">
                {cardData.map((card) => (
                    <Col key={card.id}>
                        <Card style={{ width: '18rem' }} className="h-100">
                            <Card.Body>
                                <Card.Title>{card.title}</Card.Title>
                                <Card.Text>{card.description}</Card.Text>
                            </Card.Body>
                        </Card>
                    </Col>
                ))}
            </Row>
        </Container>
    );
};

export default Hamburg;
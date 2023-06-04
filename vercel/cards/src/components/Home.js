import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import { Link } from 'react-router-dom';

const Home = () => {
    return (
        <Container className="d-flex justify-content-center mt-4">
            <Row xs={1} md={2} lg={2} className="g-4" >
                <Col >
                    <Link to="/origin/hamburg" className="text-decoration-none">
                        <Card className="h-100">
                            <Card.Body>
                                <Card.Title>Hamburg </Card.Title>
                                <Card.Text>
                                    Find out all destinations you can take
                                    from Hamburg using only your 49 Euro ticket.
                                </Card.Text>
                            </Card.Body>
                        </Card>
                    </Link>
                </Col>
                <Col >
                    <Card className="h-100">
                        <Card.Body>
                            <Card.Title>Berlin</Card.Title>
                            <Card.Text>
                                Coming soon!
                            </Card.Text>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </Container>
    );
};

export default Home;

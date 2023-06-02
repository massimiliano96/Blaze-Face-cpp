#include "Anchors.hpp"

Anchor::Anchor(float xCenter, float yCenter, float h, float w) : xCenter(xCenter), yCenter(yCenter), h(h), w(w) {};

std::string Anchor::serialize() const
{
    return "xCenter: " + std::to_string(xCenter) + ", yCenter: " + std::to_string(yCenter) + ", h: " + std::to_string(h) + ", w: " + std::to_string(w);
}

float Anchor::getX() const
{
    return xCenter;
}

float Anchor::getY() const
{
    return yCenter;
}

float Anchor::getHeight() const
{
    return h;
}

float Anchor::getWidth() const
{
    return w;
}
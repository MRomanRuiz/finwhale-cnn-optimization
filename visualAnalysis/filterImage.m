function filtered = filterImage(img)
% Function that computes the filtered image of the provided input image.
inverted = 255 - img;
filtered = 255 - (inverted - img);
end
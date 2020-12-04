vertex_shader =\
'''
#version 120

varying vec2 uv;
void main() {
    gl_Position = gl_Vertex;
    uv = vec2(gl_Vertex.x/2+0.5,gl_Vertex.y/2+0.5);
}
'''
fragment_shader =\
'''
#version 120

uniform sampler2D selectionMap;
uniform sampler2D colorMap;
uniform float wstep;
uniform float hstep;

varying vec2 uv;
void main() { 
    gl_FragColor = texture2D(colorMap,uv);
    //gl_FragColor = texture2D(selectionMap,uv);
}
'''

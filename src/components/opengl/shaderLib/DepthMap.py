vertex_shader =\
'''
#version 120
uniform mat4 worldToSrcreen;
varying float z;

void main() {
    vec4 p = worldToSrcreen * gl_Vertex;
    p /= p.w;
    gl_Position = p;
    z = gl_Position.z;
}
'''
fragment_shader =\
'''
#version 120
varying float z;
void main() {    
    gl_FragColor = vec4(z,z,z,1.0);
}
'''

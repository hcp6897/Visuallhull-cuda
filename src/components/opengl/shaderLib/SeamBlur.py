def vertex_shader(i):
    return '''
    #version 120

    varying vec2 uv;
    void main() {{
        vec4 pos = gl_Vertex;
        if({0}==0){{
            pos = vec4(gl_Vertex.x/2-0.5,gl_Vertex.y/2-0.5,0.0,1.0);
        }}else if({0}==1){{
            pos = vec4(gl_Vertex.x/2+0.5,gl_Vertex.y/2-0.5,0.0,1.0);
        }}else if({0}==2){{
            pos = vec4(gl_Vertex.x/2-0.5,gl_Vertex.y/2+0.5,0.0,1.0);
        }}else if({0}==3){{
            pos = vec4(gl_Vertex.x/2+0.5,gl_Vertex.y/2+0.5,0.0,1.0);
        }}

        gl_Position = pos;
        uv = vec2(gl_Vertex.x/2+0.5,gl_Vertex.y/2+0.5);
    }}
    '''.format(i)

def fragment_shader(i):
    return '''
    #version 120

    uniform sampler2D selectionMap;
    uniform sampler2D colorMap;
    uniform float wstep;
    uniform float hstep;

    varying vec2 uv;
    void main() {{

        vec4 centerIndex = texture2D(selectionMap,uv);

        // shrink boundary
        
        float isBoundary = 1;
        int kernelsize=7;
        for(int i=-(kernelsize-1)/2;i<=(kernelsize-1)/2;i++){{
            float isZero = 0;
            for(int j=-(kernelsize-1)/2;j<=(kernelsize-1)/2;j++){{
                vec4 neighborIndex = texture2D(selectionMap,uv+vec2(wstep*i,hstep*j));
                if((i!=0) && (j!=0) && distance(neighborIndex,centerIndex)>0 && neighborIndex.a!=0 && centerIndex.a!=0){{
                    isZero = 1;
                    isBoundary = 0;
                    break;
                }}
            }}
            if (isZero==1){{
                break;
            }}
        }}
        
        if({0}==0){{
            if(isBoundary==0){{
                int k=7;
                float count=0;
                vec4 finalColor = vec4(0.0,0.0,0.0,0.0);
                for(int i=-(k-1)/2;i<=(k-1)/2;i++){{
                    for(int j=-(k-1)/2;j<=(k-1)/2;j++){{
                        vec4 color = texture2D(colorMap,uv+vec2(wstep*i,hstep*j));
                        if(color.a!=0 && centerIndex.a!=0){{
                            finalColor += color;
                            count ++;
                        }}
                    }}
                }}
                gl_FragColor = vec4(finalColor.xyz/count,finalColor.a);
            }}else{{
                gl_FragColor = texture2D(colorMap,uv);
            }} 
        }}else if({0}==1){{
            gl_FragColor = texture2D(colorMap,uv);
        }}else if({0}==2){{
            gl_FragColor = texture2D(selectionMap,uv);
        }}else if({0}==3){{
            if(isBoundary==0){{
                gl_FragColor = vec4(1.0,0.0,0.0,1.0);
            }}else{{
                gl_FragColor = texture2D(colorMap,uv);
            }} 
        }}
    }}
    '''.format(i)

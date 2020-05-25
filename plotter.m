% %T = readtable('my_csv.csv')
% %plot(T)
% t = 0:pi/50:10*pi;
% %plot3(sin(t),cos(t),t);
% mesh(sin(t),cos(t),t);
% xlabel('sin(t)')
% 
% ylabel('cos(t)')
% zlabel('time')

delta_ = 0.5
DR = [delta_/8, delta_/6, delta_/4, delta_/3, delta_/2] ;
N = length(DR);
BS = [2,4,8,16,32]  %64,128,256,512,1024];

% t = 1:20 ;
for i = 1:N
    data(i,:) = [i+1:1:(length(BS)+i)]
end
% data = rand(N,length(BS)) ;
figure
hold on
for i = 1:N
    plot3(BS,DR,data(i,:))
end
grid on;
view(3)
xlabel('dither range')
ylabel('block size')
zlabel('accuracy')
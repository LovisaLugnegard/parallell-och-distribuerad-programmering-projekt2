figure

k=1;

while 1
  fname=['solution-' num2str(k) '.dat'];
  if ~exist(fname,'file')
    break
  end
  contents = importdata(fname);
  dims=contents(1,:);
  sol=contents(2:end,1);
  h = surf(reshape(sol,dims)');
  set(h, 'LineStyle', 'none');
  set(gca,'zlim',[-2.5 2.5])
  set(gca,'clim',[-0.2 0.2])
  title(['k = ' num2str(k)])
  pause(0.01)
  k = k + 1;
end
